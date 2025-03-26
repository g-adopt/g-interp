#!/usr/bin/env python3

### Goals for this project
### 0) interpolate a 3D mesh onto another 3D mesh in an efficient way
### 1) shared memory
###    - We want to refer to mesh objects on some controller
###      process as if they're local
### 2) streamlined MPI comms
###    - We'll likely require communication on 3 different
###      process subsets - intra-node, all controllers and
###      all
### 3) Topology awareness-ish
###    - Start by sharing memory on a single node
###      expand to other hardware units in time
import argparse
import sys
import pyvista as pv
import numpy as np

from mpi4py import MPI
from numpy.typing import NDArray
from multiprocessing import shared_memory
from pathlib import Path
from scipy.spatial import cKDTree

from typing import Callable, Union, Optional

if sys.version_info.minor < 13:
    from multiprocessing import resource_tracker


class MPI_setup:
    comm_world = MPI.COMM_WORLD
    world_rank = comm_world.Get_rank()
    world_size = comm_world.Get_size()
    ops: dict[str, Callable] = {}
    node_comm = MPI.COMM_NULL
    node_comm_rank = -1
    node_comm_size = -1
    leader_comm = MPI.COMM_NULL
    leader_comm_rank = -1
    leader_comm_size = -1

    def __init__(self):
        self.setup_mpi()

    def setup_mpi(self) -> None:
        ### Setup shared memory communicators
        self.node_comm = self.comm_world.Split_type(MPI.COMM_TYPE_SHARED)
        self.node_comm_rank = self.node_comm.Get_rank()
        self.node_comm_size = self.node_comm.Get_size()

        ### Now setup a leader comm
        leader_group = MPI.Group(self.comm_world.Get_group())
        to_gather = -1
        if self.node_comm_rank == 0:
            to_gather = self.world_rank
        leaders = list(dict.fromkeys(self.comm_world.allgather(to_gather)))
        leaders = [ r for r in leaders if r != -1 ]
        leader_group = leader_group.Incl(leaders)
        self.leader_comm = self.comm_world.Create(leader_group)
        if self.leader_comm != MPI.COMM_NULL:
            self.leader_comm_rank = self.leader_comm.Get_rank()
            self.leader_comm_size = self.leader_comm.Get_size()


class SharedMemoryHandler:
    shm = None

    def __init__(self):
        pass

    @classmethod
    def shared_memory_create(cls, size: int):
        out = cls()
        out.shm = shared_memory.SharedMemory(create=True, size=size)
        return out

    @classmethod
    def shared_memory_attach(cls, fn: str):
        out = cls()
        if sys.version_info.minor < 13:
            out.shm = shared_memory.SharedMemory(name=fn)
        else:
            out.shm = shared_memory.SharedMemory(name=fn, track=False)
        return out

    def __getattr__(self, attr):
        return getattr(self.shm, attr)

    def __del__(self):
        if sys.version_info.minor < 13:
            self.shm.close()
            try:
                self.shm.unlink()
            except FileNotFoundError:
                resource_tracker.unregister(self.shm._name, "shared_memory")
        else:
            self.shm.close()
            if self.shm._track:
                self.shm.unlink()


def get_output_grid(fn: str) -> NDArray[np.float64]:
    ### In this case, we're just going to read a new mesh
    return pv.read(fn)


class InputDataDistributor:
    y: Optional[NDArray[np.float64]] = None
    f: Optional[NDArray[np.float64]] = None
    y_shm: Optional[SharedMemoryHandler] = None
    f_shm: Optional[SharedMemoryHandler] = None

    def __init__(self, fn: Union[str, Path], mpi):
        self.mpi = mpi
        if mpi.world_rank == 0:
            self.model = pv.read(fn)
            ### We need 2 shared memory buffers, one for the grid points and the other for the data
            y_global = np.array(self.model.points)
            print("Source data read")
        else:
            y_global = np.empty(1)

        y_size, y_shape, y_dtype = mpi.comm_world.bcast((y_global.nbytes, y_global.shape, y_global.dtype), root=0)

        if mpi.leader_comm != MPI.COMM_NULL:
            self.y_shm = SharedMemoryHandler.shared_memory_create(y_size)
            self.y = np.ndarray(y_shape, dtype=y_dtype, buffer=self.y_shm.buf)
            print("Shared memory regions created")

            if mpi.leader_comm_rank == 0:
                self.y[:] = y_global[:]  # Copy data

            mpi.leader_comm.Bcast([self.y, MPI.DOUBLE], root=0)

            print("Source data distributed to remote nodes")

            ### Let everyone on our node find our shared memory location

            y_shm_name = self.y_shm.name
        else:
            y_shm_name = None
        y_shm_name = mpi.node_comm.bcast(y_shm_name, root=0)

        if mpi.node_comm_rank != 0:
            self.y_shm = SharedMemoryHandler.shared_memory_attach(y_shm_name)
            self.y = np.ndarray(y_shape, dtype=y_dtype, buffer=self.y_shm.buf)

            print("Shared memory regions attached")

    def distribute_field(self, field: str):
        if self.mpi.world_rank == 0:
            f_global = np.array(self.model[field])
        else:
            f_global = np.empty(1)
        f_size, f_shape, f_dtype = self.mpi.comm_world.bcast((f_global.nbytes, f_global.shape, f_global.dtype), root=0)
        if self.mpi.leader_comm != MPI.COMM_NULL:
            if self.f_shm is not None:
                if self.f_shm.size != f_size:
                    del self.f_shm
            if self.f_shm is None:
                self.f_shm = SharedMemoryHandler.shared_memory_create(f_size)
                self.f = np.ndarray(f_shape, dtype=f_dtype, buffer=self.f_shm.buf)

            if self.mpi.leader_comm_rank == 0:
                np.copyto(self.f, f_global, casting="no")
            self.mpi.leader_comm.Bcast([self.f, MPI.DOUBLE], root=0)
            f_shm_name = self.f_shm.name
        else:
            f_shm_name = None
        f_shm_name = self.mpi.node_comm.bcast(f_shm_name, root=0)

        if self.mpi.node_comm_rank != 0:
            if self.f_shm is not None:
                if self.f_shm.name != f_shm_name:
                    del self.f_shm
            if self.f_shm is None:
                self.f_shm = SharedMemoryHandler.shared_memory_attach(f_shm_name)
                self.f = np.ndarray(f_shape, dtype=f_dtype, buffer=self.f_shm.buf)


class Interpolator:
    epsilon_distance = 1e-8

    def __init__(
        self,
        source_grid: NDArray[np.float64],
        leafsize: int = 16,
        nneighbours: int = 16,
    ):
        self.tree = cKDTree(data=source_grid, leafsize=leafsize)
        self.nneighbours = nneighbours

    def __call__(self, target_coords: NDArray[np.float64], data: NDArray[np.float64]):
        dists, idx = self.tree.query(x=target_coords, k=self.nneighbours)
        close_points_mask = dists[:, 0] < self.epsilon_distance

        # Use np.where to avoid division by very small values
        # Replace tiny distances with self.epsilon_distance to avoid division
        # by tiny values while keeping original distances intact for later use
        safe_dists = np.where(dists < self.epsilon_distance, self.epsilon_distance, dists)

        # Then, calculate the weighted average using safe_dists
        weights = self.kernel_weights(safe_dists)
        if len(data.shape) == 2:
            weights = weights[:, :, np.newaxis]
        interped = np.sum(weights * data[idx], axis=1)

        # Now handle the case where points are too close to each other:
        interped[close_points_mask] = data[idx[close_points_mask, 0]]
        return interped

    def kernel_weights(self, dist: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.normalise(1 / dist)

    def normalise(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.einsum("i, ij->ij", 1 / np.sum(weights, axis=1), weights)


def main():
    parser = argparse.ArgumentParser(
        prog="mpi-interp", description="Interpolates a large pvtu file onto a smaller grid"
    )
    parser.add_argument("-i", "--infile", required=True, help="The input file to interpolate")
    parser.add_argument(
        "-g", "--input_grid", required=True, help="The input file containing the grid to interpolate to"
    )
    parser.add_argument("-o", "--outfile", required=True, help="Path to the final output")
    parser.add_argument("-f", "--field", required=False, help="Name(s) of field to be interpolated, comma separated")

    ns = parser.parse_args(sys.argv[1:])

    mpi = MPI_setup()
    input_data = InputDataDistributor(ns.infile, mpi)
    if ns.field:
        fields = ns.field.split(",")
    else:
        if mpi.world_rank == 0:
            fields = input_data.model.array_names
        else:
            fields = []
        fields = mpi.comm_world.bcast(fields, root=0)
    interp = Interpolator(input_data.y, nneighbours=32)
    print("Interpolant constructed")

    if mpi.world_rank == 0:
        out_model = get_output_grid(ns.input_grid)
        out_grid = np.array(out_model.points)
        downscaled = pv.UnstructuredGrid()
        downscaled.copy_structure(out_model)
        # Distribute outgrid points
        reqs = []
        subgrid_size = len(out_grid) // mpi.world_size
        remainder = len(out_grid) % mpi.world_size
        my_subgrid_size = subgrid_size + (1 if remainder > 0 else 0)
        subgrid = out_grid[:my_subgrid_size]
        end = my_subgrid_size
        for i in range(1, mpi.world_size):
            start = end
            end = start + subgrid_size + (1 if i < remainder else 0)
            reqs.append(mpi.comm_world.isend(out_grid[start:end], dest=i, tag=99))
        _ = MPI.Request.waitall(reqs)
    else:
        subgrid = mpi.comm_world.recv(source=0, tag=99)
    print("Target grid distributed")

    for field in fields:
        input_data.distribute_field(field)
        out_f_part = interp(subgrid, input_data.f)
        out_f_parts = mpi.comm_world.gather(out_f_part, root=0)

        ### Rank 0 from here on out
        if mpi.world_rank == 0:
            out_f = np.empty([len(out_grid)] + list(input_data.model[field].shape[1:]), dtype=np.float64)
            end = 0
            for i, part in enumerate(out_f_parts):
                start = end
                end = start + subgrid_size + (1 if i < remainder else 0)
                out_f[start:end] = part

            downscaled[field] = out_f

        mpi.comm_world.Barrier()

    if mpi.world_rank == 0:
        downscaled.save(ns.outfile)
        print("=================================================================")

    del input_data


if __name__ == "__main__":
    main()
