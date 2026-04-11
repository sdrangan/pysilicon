---
title: Extracting VCD Files
parent: Timing Analysis Tools
nav_order: 2
has_children: false
---

# Extracting VCD Files

The Vitis C/RTL simulation creates a `.wdb` (waveform database)  file with traces of all the signals.  This file is in an AMD proprietary format and cannot be read by other programs,
although you can see it in the Vivado viewer.  PySilicon provides methods to modify the simulation to export an alternative open-source **VCD** or [**Value Change Dump**](https://en.wikipedia.org/wiki/Value_change_dump) format.  VCD files can be read by many programs including python.

## Pre-Requisites

Prior to capturing the VCD file, you must first run the C/RTL co-simulation of the design.  To run co-simulation from a TCL file, use the command:

```tcl
cosim_design -trace_level [all | port]
```

It is important to set the `trace_level` parameter as this will tell Vitis to trace signals.

## Parameters

Next, you will have to identify the following parameters.

| Parameter    | Default           | Description |
|--------------|-------------------|-------------|
| `top`        | *(required)*      | Top-level function name |
| `comp`       | `"hls_component"` | HLS component directory name |
| `out`        | `"dump.vcd"`      | Output filename (written to `vcd/<out>`) |
| `soln`       | `"solution1"`     | Solution subdirectory inside `comp` |
| `trace_level`| `"*"`             | VCD trace level: `"*"` for all signals, `"port"` for port signals only |
| `workdir`    | CWD               | Working directory containing the `comp` folder |


Note that you can get the component and solution name from the directory structure created by Vitis.  Specifically, running Vitis RTL co-sim creates directories of the form:  `<comp>/<soln>`.   For example, in the [polynomial example](../../examples/poly/), the RTL simulation generates a directory:

```bash
pysilicon_poly_proj/solution1
```

So, `comp` is `pysilicon_hist_proj` and `soln` is `solution1`.   


For the value of `top`, this parameter should match the value of `top` used in the RTL simulation.  For example, in the polynomial example, the TCL file has the command `set_top poly`, so `top` is `poly`.


## Generating the VCD from the CLI

To then generate the VCD file from the CLI, first activate the virtual environment where pysilicon is installed.  Then run:

```bash
(env) xsim_vcd --top <top> [--comp <comp>] [--soln <soln> ] [--out <out>]
```

So, in the polynomial example this would be:

```bash
(env) xsim_vcd --top pysilicon_poly_proj --comp solution1 --out dump_poly.vcd
```
After running the script, there will be a VCD file with the simulation.  In the example above, it will be 

```bash
poly/vcd/dump_poly.vcd
```

## Python API

You can also call the function from the `run_xsim_vcd` function that wraps the same logic as the CLI entry point:

```python
from pysilicon.scripts.xsim_vcd import run_xsim_vcd
from pathlib import Path

vcd_path = run_xsim_vcd(
    top="poly",
    comp="pysilicon_poly_proj",
    out="dump.vcd",
    workdir=Path("examples/poly"),
)
print(f"VCD written to: {vcd_path}")
```


The `run_xsim_vcd` function returns a `pathlib.Path` pointing to the written VCD file.  It raises the following errors:

- Raises `RuntimeError` on non-Windows platforms
- Raises `FileNotFoundError` if the simulation directory is not found
- Raises `RuntimeError` if xsim fails

## Understanding the `xsim_vcd.py` script.

The script  `xsim_vcd.py` is used to automate the process of adding a VCD trace.  But you may want to know how this function works, in case you need to modify later.  Basically, the `xsim_vcd.py` does these steps automatically:

- After running the initial simulation, it locates the directory where the simulation files are.
For the scalar adder simulation, it will be in something like:

```bash
scalar_fun_vitis\hls_component\scalar_fun\hls\sim\verilog
```

This large directory contains automatically generated RTL files for the testbench along with simuation files. We will modify these files to output a VCD file and re-run the simulation. 
- In this directory, there will be a file `scalar_fun.tcl` which sets the configuration for the simulation.  Copy the file to a new file `scalar_fun.tcl` and modify as follows:
   - Add initial lines at the top of the file (before the `log_wave -r /`) line:

```tcl
open_vcd
log_vcd * 
```

    - At the eend of the file there are lines:

```tcl
run all
quit
```

    These lines are changed to:

```
run all
close_vcd
quit
```

- In the same directory, there is a file, `run_xsim.bat`.  
   - There should be a line like:

```bash
call C:/Xilinx/2025.1/Vivado/bin/xsim  ... -tclbatch scalar_fun.tcl -view add_dataflow_ana.wcfg -protoinst add.protoinst
```
   
   - The script `xsim_vcd` copies just this line to a new file `run_xsim_vcd.bat` and modifies that line to:

```bash
cd /d "%~dp0"
call C:/Xilinx/2025.1/Vivado/bin/xsim  ... -tclbatch scalar_fun_vcd.tcl -view add_dataflow_ana.wcfg -protoinst add.protoinst
```

    That is, the script add a `cd /d` command to make the file callable from a different directory, and the script changes the `tclbatch` file from `scalar_fun.tcl` to `scalar_fun_vcd.tcl`
- Go back to the directory `scalar_fun_vitis` Re-run the simulation with 

```bash
./run_xsim_vcd.bat
```

This will re-run the simulation and create a `dump.vcd` file of the simulation data.
