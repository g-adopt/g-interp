# G-Interp
 Post processing utilities for G-ADOPT model output

## Installation
Create a Python virtual environment, and install the dependencies into it:

```sh
python3 -m venv env
source env/bin/activate
python3 -m pip install git+https://github.com/g-adopt/g-interp.git
```

### vtk-interp
Interpolate unstructured VTK files onto a regular lat/lon grid

Run `vtk-interp --help` to get the help for the command-line utility.

### mpi-interp
Interpolate one or more unstructured fields from a VTK file to the grid provided from another VTK file. Can be run in parallel to interpolate very large datasets.

Run `mpi-interp --help` to get the help for the command-line utility.