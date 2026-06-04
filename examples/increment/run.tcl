open_project -reset pysilicon_incr_proj
set_top incr
# Kernel source comes from gen/ (HlsCodegenStep output).  The non-templated
# transform hook body lives in incr_transform_impl.cpp at the source-tree root;
# the templated respond hook body is #include'd by gen/incr.hpp from
# "../incr_respond_impl.tpp".
add_files gen/incr.cpp -cflags "-I."
add_files incr_transform_impl.cpp -cflags "-I."
add_files -tb incr_tb.cpp -cflags "-I."

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
if {[info exists ::env(PYSILICON_INCR_CLK_PERIOD_NS)]} {
    set clk_period_ns $::env(PYSILICON_INCR_CLK_PERIOD_NS)
}
create_clock -period $clk_period_ns
set data_dir [file join $script_dir "data"]

set do_cosim 0
if {[info exists ::env(PYSILICON_INCR_COSIM)]} {
    set do_cosim [expr {$::env(PYSILICON_INCR_COSIM) in {1 true TRUE yes YES}}]
}

if {[catch {csim_design -argv "$data_dir"} res]} {
    puts "PYSILICON_ERROR: incr C-Simulation failed."
    puts $res
    exit 1
}

if {$do_cosim} {
    if {[catch {csynth_design} res]} {
        puts "PYSILICON_ERROR: incr C-Synthesis failed."
        puts $res
        exit 1
    }
    if {[catch {cosim_design -argv "$data_dir"} res]} {
        puts "PYSILICON_ERROR: incr RTL Co-Simulation failed."
        puts $res
        exit 1
    }
    puts "PYSILICON_SUCCESS: incr C-Simulation, C-Synthesis, and RTL Co-Simulation passed."
} else {
    puts "PYSILICON_SUCCESS: incr C-Simulation passed."
}
exit 0
