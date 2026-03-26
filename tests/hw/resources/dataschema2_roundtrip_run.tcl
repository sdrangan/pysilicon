open_project -reset pysilicon_dataschema2_vitis_proj
set_top main
add_files -tb roundtrip_test.cpp
open_solution -reset "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10

set script_dir [file dirname [file normalize [info script]]]
set in_words_path [file join $script_dir "packet_words.txt"]
set out_words_path [file join $script_dir "packet_words_out.txt"]

if {[catch {csim_design -argv "$in_words_path $out_words_path"} res]} {
    puts "PYSILICON_ERROR: HLS C-Simulation failed."
    puts $res
    exit 1
}
puts "PYSILICON_SUCCESS: Dataschema2 Vitis roundtrip test passed."
exit 0
