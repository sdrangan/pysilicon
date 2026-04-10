---
title: Extracting VCD Files
parent: Timing Analysis Tools
nav_order: 2
has_children: false
---

# Extracting VCD Files

The Vitis C/RTL simulation creates a `.wdb` (waveform database)  file with traces of all the signals.  This file is in an AMD proprietary format and cannot be read by other programs,
although you can see it in the Vivado viewer.  PySilicon provides methods to modify the simulation to export an alternative open-source **VCD** or [**Value Change Dump**](https://en.wikipedia.org/wiki/Value_change_dump) format.  VCD files can be read by many programs including python.

## Generating VCD Files from the command line

To capture a VCD file:

- First run the C/RTL co-simulation of the design.  To run co-simulation from a TCL file, use the command:

```tcl
cosim_design -trace_level [all/port]
```

It is important to set the `trace_level` parameter as this will tell Vitis to trace signals:
- After the RTL co-simulation is completed, [activate the virtual environment](../installation/python.md) for PySilicon package
- Identify the `component_name` and `top_name` of the project.  For example, in the histogram example in `hwdesign/examples/histogram`, the `component_name` is `pysilicon_hist_proj` and the `top_name` is `solution1`.  
- Re-run the simulation with VCD with the command from PowerShell or Linux terminal:

```bash
(env) xsim_vcd --top <top_name> [--comp <component_name>] [--out <vcd_file>]
```

where `vcdfile` is the name of the VCD file with the signal traces.  By default, `<vcd_file>` is `dump.vcd`.  So, in the case of the histogram example, you would navigate to `hwdesign/examples/histogram` and then run:

```bash
(env) xsim_vcd --top pysilicon_hist_proj --comp solution1 --out dump_hist.vcd
```

- Note:  We have not yet created a version of the script `xsim_vcd` for Linux.
- After running the script, there will be a VCD file with the simulation.  In the example above, it will be 

```bash
histogram/dump_hist.vcd
```

## Viewing the Timing Diagram
After you have created VCD file, you can see the timing diagram from the [jupyter notebook](https://github.com/sdrangan/hwdesign/tree/main/scalar_fun/notebooks/view_timing.ipynb).

## Understanding the `xsim_vcd.py` function. 

I wrote the function  `xsim_vcd.py` to automate the process of adding a VCD trace.
But you may want to know how this function works, in case you need to modify later.
Basically, the `xsim_vcd.py` does these steps automatically.

* After running the initial simulation, locate the directory where the simulation files are.
For the scalar adder simulation, it will be in something like:

```bash
scalar_fun_vitis\hls_component\scalar_fun\hls\sim\verilog
```

This large directory contains automatically generated RTL files for the testbench along with simuation files.
We will modify these files to output a VCD file and re-run the simulation. 
* In this directory, there will be a file `scalar_fun.tcl` which sets the configuration for the simulation.  Copy the file to a new file `scalar_fun.tcl` and modify as follows:
   *  Add initial lines at the top of the file (before the `log_wave -r /`) line:

    ```tcl
    open_vcd
    log_vcd * 
    ```

    * At the eend of the file there is:
    ```tcl
    run all
    quit
    ```
    
    Modify these lines to:
    ```
    run all
    close_vcd
    quit
    ```

* In the same directory, there is a file, `run_xsim.bat`.  
   * There should be a line like:

    ```bash
    call C:/Xilinx/2025.1/Vivado/bin/xsim  ... -tclbatch scalar_fun.tcl -view add_dataflow_ana.wcfg -protoinst add.protoinst
    ```
   
   * Copy just this line to a new file `run_xsim_vcd.bat` and modify that line to:

    ```bash
    cd /d "%~dp0"
    call C:/Xilinx/2025.1/Vivado/bin/xsim  ... -tclbatch scalar_fun_vcd.tcl -view add_dataflow_ana.wcfg -protoinst add.protoinst
    ```

    That is, we add a `cd /d` command to make the file callable from a different directory, and we change the `tclbatch` file from `scalar_fun.tcl` to `scalar_fun_vcd.tcl`
* Go back to the directory `scalar_fun_vitis` Re-run the simulation with 

```bash
./run_xsim_vcd.bat
```

This will re-run the simulation and create a `dump.vcd` file of the simulation data.
