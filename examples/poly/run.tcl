open_project -reset pysilicon_poly_proj
set_top poly
add_files poly.cpp
add_files -tb poly_tb.cpp

open_solution -reset "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10

set script_dir [file dirname [file normalize [info script]]]
set data_dir [file join $script_dir "data"]

if {[catch {csim_design -argv "$data_dir"} res]} {
    puts "PYSILICON_ERROR: poly C-Simulation failed."
    puts $res
    exit 1
}

puts "PYSILICON_SUCCESS: poly C-Simulation passed."
exit 0
