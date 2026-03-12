#include <fstream>
#include <iostream>

__EXTRA_INCLUDES__
#include "__HEADER__"

int main(int argc, char** argv) {
    const char* in_json_path = (argc > 1) ? argv[1] : "packet_src.json";
    const char* out_words_path = (argc > 2) ? argv[2] : "packet_from_vitis_words.txt";

    __PACKET_CLASS__ pkt;
    try {
        pkt.load_json_file(in_json_path);
    } catch (const std::exception& ex) {
        std::cerr << "Failed to load JSON: " << ex.what() << std::endl;
        return 1;
    }

    ap_uint<__WORD_BW__> words[__NWORDS__];
    pkt.write_array<__WORD_BW__>(words__RW_ARGS__);

    std::ofstream out_words(out_words_path);
    if (!out_words) {
        std::cerr << "Failed to open output words file: " << out_words_path << std::endl;
        return 1;
    }

    for (int i = 0; i < __NWORDS__; ++i) {
        out_words << static_cast<unsigned long long>(words[i]) << "\n";
    }

    return 0;
}
