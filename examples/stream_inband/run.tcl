open_project -reset waveflow_poly_proj
set_top poly
# Kernel sources come from gen/ (HlsCodegenStep output).  The hand-written
# evaluate body lives in poly_evaluate_impl.tpp at the source-tree root and
# is #include'd from gen/poly.hpp via "../poly_evaluate_impl.tpp".
add_files gen/poly.cpp -cflags "-I."
add_files -tb gen/poly_tb.cpp -cflags "-I."

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
set clk_period_ns 10
if {[info exists ::env(WAVEFLOW_POLY_CLK_PERIOD_NS)]} {
    set clk_period_ns $::env(WAVEFLOW_POLY_CLK_PERIOD_NS)
}
create_clock -period $clk_period_ns
set data_dir [file join $script_dir "data"]

set do_cosim 0
set trace_level "none"

if {[info exists ::env(WAVEFLOW_POLY_COSIM)]} {
    set do_cosim [expr {$::env(WAVEFLOW_POLY_COSIM) in {1 true TRUE yes YES}}]
}
if {[info exists ::env(WAVEFLOW_POLY_TRACE_LEVEL)]} {
    set trace_level $::env(WAVEFLOW_POLY_TRACE_LEVEL)
}

for {set i 0} {$i < [llength $argv]} {incr i} {
    set arg [lindex $argv $i]
    if {$arg eq "--cosim"} {
        set do_cosim 1
    } elseif {$arg eq "--trace-level"} {
        incr i
        if {$i >= [llength $argv]} {
            puts "WAVEFLOW_ERROR: Missing value after --trace-level."
            exit 1
        }
        set trace_level [lindex $argv $i]
    }
}

if {$trace_level ni {none port all}} {
    puts "WAVEFLOW_ERROR: Unsupported trace level '$trace_level'. Expected one of: none, port, all."
    exit 1
}

if {[catch {csim_design -argv "$data_dir"} res]} {
    puts "WAVEFLOW_ERROR: poly C-Simulation failed."
    puts $res
    exit 1
}

if {$do_cosim} {
    if {[catch {csynth_design} res]} {
        puts "WAVEFLOW_ERROR: poly C-Synthesis failed."
        puts $res
        exit 1
    }

    if {[catch {cosim_design -argv "$data_dir" -trace_level $trace_level} res]} {
        puts "WAVEFLOW_ERROR: poly RTL Co-Simulation failed."
        puts $res
        exit 1
    }

    puts "WAVEFLOW_SUCCESS: poly C-Simulation, C-Synthesis, and RTL Co-Simulation passed."
} else {
    if {[catch {csynth_design} res]} {
        puts "WAVEFLOW_ERROR: poly C-Synthesis failed."
        puts $res
        exit 1
    }

    puts "WAVEFLOW_SUCCESS: poly C-Simulation and C-Synthesis passed."
}
exit 0
