# smoke_test.tcl
# Setup the HLS project environment
open_project -reset pysilicon_smoke_proj

# The 'main' function in our C++ file is the entry point for the testbench
set_top main
add_files -tb main.cpp

# Define the target hardware and clock
open_solution -reset "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10

# Execute C-Simulation
# We wrap this in a catch block so we can exit gracefully on failure
if {[catch {csim_design} res]} {
    puts "PYSILICON_ERROR: HLS C-Simulation failed."
    puts $res
    exit 1
}

puts "PYSILICON_SUCCESS: HLS environment is functional."
exit 0