#include <iostream>

__EXTRA_INCLUDES__
#include "__HEADER__"

int main(int argc, char** argv) {
    const char* in_bin_path = (argc > 1) ? argv[1] : "packet_words.bin";
    const char* out_json_path = (argc > 2) ? argv[2] : "packet_out.json";

    __PACKET_CLASS__ pkt = {};
    try {
        __READ_CALL__;
        pkt.dump_json_file(out_json_path, 4);
    } catch (const std::exception& ex) {
        std::cerr << "Loopback uint32 file test failed: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
