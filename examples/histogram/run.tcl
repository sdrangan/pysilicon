open_project -reset pysilicon_hist_proj
set_top hist
add_files hist.cpp
add_files -tb hist_tb.cpp

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

if {[catch {csim_design -argv "$data_dir"} res]} {
    puts "PYSILICON_ERROR: hist C-Simulation failed."
    puts $res
    exit 1
}

# Pass --cosim via --tclargs to also run C-synthesis and RTL co-simulation.
set do_cosim 0
if {[lsearch $argv "--cosim"] >= 0} {
    set do_cosim 1
}

if {$do_cosim} {
    if {[catch {csynth_design} res]} {
        puts "PYSILICON_ERROR: hist C-Synthesis failed."
        puts $res
        exit 1
    }
    if {[catch {cosim_design -argv "$data_dir"} res]} {
        puts "PYSILICON_ERROR: hist RTL Co-Simulation failed."
        puts $res
        exit 1
    }
    puts "PYSILICON_SUCCESS: hist C-Simulation, C-Synthesis, and RTL Co-Simulation passed."
} else {
    puts "PYSILICON_SUCCESS: hist C-Simulation passed."
}
exit 0
