# Vitis HLS C-simulation driver for one basic_vec MAC kernel.
# Uniform argv: in_a in_b in_c out_bits.  -ffp-contract=off keeps the float kernel's
# a*b+c as two roundings (matching numpy float32), never a fused FMA.
open_project -reset basic_vec_proj
set_top main
add_files -tb kernel.cpp -cflags "-ffp-contract=off"

open_solution -reset "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10

set d [file dirname [file normalize [info script]]]
set argv_paths "[file join $d in_a.txt] [file join $d in_b.txt] [file join $d in_c.txt] [file join $d out_bits.txt]"

if {[catch {csim_design -argv $argv_paths} res]} {
    puts "WAVEFLOW_ERROR: HLS C-Simulation failed."
    puts $res
    exit 1
}
puts "WAVEFLOW_SUCCESS: basic_vec csim passed."
exit 0
