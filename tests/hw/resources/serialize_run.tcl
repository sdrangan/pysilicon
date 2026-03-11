open_project -reset pysilicon_dataschema_vitis_proj
set_top main
add_files -tb serialize_test.cpp
open_solution -reset "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10

set script_dir [file dirname [file normalize [info script]]]
set words_path [file join $script_dir "packet_words.txt"]
set out_json_path [file join $script_dir "packet_out.json"]

if {[catch {csim_design -argv "$words_path $out_json_path"} res]} {
    # Keep previous catch semantics for diagnostics.
    puts "PYSILICON_ERROR: HLS C-Simulation failed."
    puts $res
    exit 1
}
puts "PYSILICON_SUCCESS: Dataschema Vitis serialization test passed."
exit 0
