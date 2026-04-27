# Source: examples/poly/run.tcl

Original extension: `.tcl`

```tcl
open_project -reset pysilicon_poly_proj
set_top poly
add_files poly.cpp
add_files -tb poly_tb.cpp

set script_dir [file dirname [file normalize [info script]]]
set streamutils_cpp [file join $script_dir "streamutils.cpp"]
if {![file exists $streamutils_cpp]} {
    set streamutils_cpp [file join $script_dir "include" "streamutils.cpp"]
}
if {[file exists $streamutils_cpp]} {
    add_files -tb $streamutils_cpp
}

open_solution -reset "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10
set data_dir [file join $script_dir "data"]

set do_cosim 0
set trace_level "none"

if {[info exists ::env(PYSILICON_POLY_COSIM)]} {
    set do_cosim [expr {$::env(PYSILICON_POLY_COSIM) in {1 true TRUE yes YES}}]
}
if {[info exists ::env(PYSILICON_POLY_TRACE_LEVEL)]} {
    set trace_level $::env(PYSILICON_POLY_TRACE_LEVEL)
}

for {set i 0} {$i < [llength $argv]} {incr i} {
    set arg [lindex $argv $i]
    if {$arg eq "--cosim"} {
        set do_cosim 1
    } elseif {$arg eq "--trace-level"} {
        incr i
        if {$i >= [llength $argv]} {
            puts "PYSILICON_ERROR: Missing value after --trace-level."
            exit 1
        }
        set trace_level [lindex $argv $i]
    }
}

if {$trace_level ni {none port all}} {
    puts "PYSILICON_ERROR: Unsupported trace level '$trace_level'. Expected one of: none, port, all."
    exit 1
}

if {[catch {csim_design -argv "$data_dir"} res]} {
    puts "PYSILICON_ERROR: poly C-Simulation failed."
    puts $res
    exit 1
}

if {$do_cosim} {
    if {[catch {csynth_design} res]} {
        puts "PYSILICON_ERROR: poly C-Synthesis failed."
        puts $res
        exit 1
    }

    if {[catch {cosim_design -argv "$data_dir" -trace_level $trace_level} res]} {
        puts "PYSILICON_ERROR: poly RTL Co-Simulation failed."
        puts $res
        exit 1
    }

    puts "PYSILICON_SUCCESS: poly C-Simulation, C-Synthesis, and RTL Co-Simulation passed."
} else {
    if {[catch {csynth_design} res]} {
        puts "PYSILICON_ERROR: poly C-Synthesis failed."
        puts $res
        exit 1
    }

    puts "PYSILICON_SUCCESS: poly C-Simulation and C-Synthesis passed."
}
exit 0

```
