open_project -reset pysilicon_dataschema_uint32_file_proj
set_top main
add_files -tb uint32_file_read_test.cpp

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
set in_bin_path [file join $script_dir "packet_words.bin"]
set out_json_path [file join $script_dir "packet_out.json"]

if {[catch {csim_design -argv "$in_bin_path $out_json_path"} res]} {
    puts "PYSILICON_ERROR: HLS C-Simulation failed."
    puts $res
    exit 1
}
puts "PYSILICON_SUCCESS: Dataschema uint32 file loopback test passed."
exit 0
