# run.tcl
# Vitis HLS C-simulation script for poly DUT + poly_tb testbench.

open_project -reset pysilicon_poly_proj

# DUT top-level function.
set_top poly

# Design and testbench sources.
add_files poly.cpp
add_files -tb poly_tb.cpp

open_solution -reset "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10

# Run C simulation from this script directory so relative data files in
# poly_tb.cpp (cmd_hdr_data.bin, samp_in_data.bin, etc.) resolve correctly.
set script_dir [file dirname [file normalize [info script]]]
cd $script_dir

if {[catch {csim_design} res]} {
    puts "PYSILICON_ERROR: poly C-Simulation failed."
    puts $res
    exit 1
}

puts "PYSILICON_SUCCESS: poly C-Simulation passed."
exit 0
