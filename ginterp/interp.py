import argparse

import numpy as np
import pyvista as pv
import xarray as xr


def interpolate(filename: str, *, radii: tuple[float, float], cylindrical: bool, dims: tuple[int, ...]):
    if cylindrical:
        nx, nz = dims
        ny = 1
        grid_shape = nz, nx
    else:
        nx, ny, nz = dims
        grid_shape = nz, nx, ny

    r_min, r_max = radii

    theta = np.linspace(0, 360, nx + 1)[:-1]
    phi = np.linspace(0, 180, ny)
    r = np.linspace(r_min, r_max, nz)

    x, y, z = np.meshgrid(np.radians(theta), 0 if cylindrical else np.radians(phi), r)
    phi_factor = 1 if cylindrical else np.sin(y)

    x_cart = z * np.cos(x) * phi_factor
    y_cart = z * np.sin(x) * phi_factor

    if cylindrical:
        z_cart = y
    else:
        z_cart = z * np.cos(y)

    cylinder_grid = pv.StructuredGrid(x_cart, y_cart, z_cart)

    g = pv.read(filename)
    # either interpolate for weighted, or sample for direct
    sampled = cylinder_grid.sample(g)

    coords = {"lon": theta, "depth": r}
    if cylindrical:
        coord_list = ("depth", "lon")
    else:
        coord_list = ("depth", "lon", "lat")
        coords["lat"] = 90 - phi

    ds = xr.Dataset(coords=coords)

    for name, data in sampled.point_data.items():
        # skip info arrays
        if name.startswith("vtk") and name != "vtkValidPointMask":
            continue

        if len(data.shape) > 1:
            # vector data
            for suffix, component in zip("xyz", data.T):
                ds[f"{name}_{suffix}"] = (coord_list, component.reshape(grid_shape))

        else:
            ds[name] = (coord_list, data.reshape(grid_shape))

    ds["lon"].attrs["units"] = "degrees_east"
    ds["lon"].attrs["axis"] = "X"
    ds["depth"].attrs["axis"] = "Z"
    ds["depth"].attrs["positive"] = "up"

    if not cylindrical:
        ds["lat"].attrs["units"] = "degrees_north"
        ds["lat"].attrs["axis"] = "Y"

    return ds


def main():
    parser = argparse.ArgumentParser(
        prog="vtk-interp",
        description="Interpolate from unstructured VTK grids to netCDF",
    )
    parser.add_argument("filename")
    parser.add_argument(
        "-r",
        "--radii",
        required=True,
        help="Comma-separated minimum and maximum radii for the grid",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="interp.nc",
        help="Output filename, defaults to interp.nc",
    )
    parser.add_argument(
        "-d",
        "--dims",
        required=True,
        help="Comma-separated list of dimensions; cylindrical: (nx,nz) or spherical: (nx,ny,nz)",
    )

    grid_group = parser.add_mutually_exclusive_group(required=True)
    grid_group.add_argument(
        "-c",
        "--cylindrical",
        action="store_true",
        help="Grid is a cylindrical annulus (2D)",
    )
    grid_group.add_argument(
        "-s",
        "--spherical",
        action="store_false",
        help="Grid is 3D spherical",
    )

    args = parser.parse_args()

    radii = [float(s) for s in args.radii.split(",")]
    dims = [int(s) for s in args.dims.split(",")]

    ds = interpolate(args.filename, radii=radii, cylindrical=args.cylindrical, dims=dims)
    ds.to_netcdf(args.output)


if __name__ == "__main__":
    main()
