open_project -reset pysilicon_dataunion_loopback_proj
set_top main
add_files -tb dataunion_loopback_test.cpp

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
set in_path  [file join $script_dir "dataunion_words_in.txt"]
set out_path [file join $script_dir "dataunion_words_out.txt"]

if {[catch {csim_design -argv "$in_path $out_path"} res]} {
    puts "PYSILICON_ERROR: DataUnion loopback C-Simulation failed."
    puts $res
    exit 1
}
puts "PYSILICON_SUCCESS: DataUnion loopback test passed."
exit 0
