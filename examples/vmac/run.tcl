# Vitis HLS C-simulation driver for one VMAC conformance kernel.
# The generated kernel.cpp + in_a.txt (the shared-memory image) are written next to this
# script (each case runs in its own directory); csim writes out_bits.txt back here.
# Uniform argv: in_a in_b out_bits  (in_a = mem image; in_b is unused).
open_project -reset vmac_conf_proj
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
puts "WAVEFLOW_SUCCESS: vmac conformance csim passed."
exit 0
