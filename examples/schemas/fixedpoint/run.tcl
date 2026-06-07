# Vitis HLS C-simulation driver for one fixed-point conformance kernel.
# The generated kernel.cpp + in_a.txt / in_b.txt are written next to this script
# (each config runs in its own directory); csim writes out_bits.txt back here.
# Uniform argv: in_a in_b out_bits (requant ignores in_b).
open_project -reset fixedpoint_conf_proj
set_top main
add_files -tb kernel.cpp

open_solution -reset "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10

set d [file dirname [file normalize [info script]]]
set argv_paths "[file join $d in_a.txt] [file join $d in_b.txt] [file join $d out_bits.txt]"

if {[catch {csim_design -argv $argv_paths} res]} {
    puts "WAVEFLOW_ERROR: HLS C-Simulation failed."
    puts $res
    exit 1
}
puts "WAVEFLOW_SUCCESS: fixedpoint conformance csim passed."
exit 0
