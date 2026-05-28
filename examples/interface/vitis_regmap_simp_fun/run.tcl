open_project -reset pysilicon_simp_fun_proj
set_top simp_fun
add_files gen/simp_fun.cpp -cflags "-I."
add_files simp_fun_compute_impl.cpp -cflags "-I."
add_files -tb gen/simp_fun_tb.cpp -cflags "-I."

set script_dir [file dirname [file normalize [info script]]]
open_solution -reset "solution1"
set_part {xc7z020clg484-1}
set clk_period_ns 10
if {[info exists ::env(PYSILICON_SIMP_FUN_CLK_PERIOD_NS)]} {
    set clk_period_ns $::env(PYSILICON_SIMP_FUN_CLK_PERIOD_NS)
}
create_clock -period $clk_period_ns
set data_dir [file join $script_dir "data"]

set do_cosim 0
set trace_level "none"
if {[info exists ::env(PYSILICON_SIMP_FUN_COSIM)]} {
    set do_cosim [expr {$::env(PYSILICON_SIMP_FUN_COSIM) in {1 true TRUE yes YES}}]
}
if {[info exists ::env(PYSILICON_SIMP_FUN_TRACE_LEVEL)]} {
    set trace_level $::env(PYSILICON_SIMP_FUN_TRACE_LEVEL)
}

if {[catch {csim_design -argv "$data_dir"} res]} {
    puts "PYSILICON_ERROR: simp_fun C-simulation failed."
    puts $res
    exit 1
}

if {[catch {csynth_design} res]} {
    puts "PYSILICON_ERROR: simp_fun C-synthesis failed."
    puts $res
    exit 1
}

if {$do_cosim} {
    if {[catch {cosim_design -argv "$data_dir" -trace_level $trace_level} res]} {
        puts "PYSILICON_ERROR: simp_fun RTL co-simulation failed."
        puts $res
        exit 1
    }
    puts "PYSILICON_SUCCESS: simp_fun C-sim, C-synth, and RTL co-sim passed."
} else {
    puts "PYSILICON_SUCCESS: simp_fun C-sim and C-synth passed."
}
exit 0
