#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

__EXTRA_INCLUDES__
#include "__HEADER__"

int main(int argc, char** argv) {
    const char* in_words_path = (argc > 1) ? argv[1] : "packet_words.txt";
    const char* out_json_path = (argc > 2) ? argv[2] : "packet_out.json";

    std::ifstream in_words(in_words_path);
    if (!in_words) {
        std::cerr << "Failed to open input words file: " << in_words_path << std::endl;
        return 1;
    }

    std::vector<ap_uint<__WORD_BW__>> words;
    unsigned long long raw = 0;
    while (in_words >> raw) {
        words.push_back((ap_uint<__WORD_BW__>)raw);
    }

    if (words.empty()) {
        std::cerr << "No serialized words found." << std::endl;
        return 1;
    }

    __PACKET_CLASS__ pkt;
    pkt.read_array<__WORD_BW__>(words.data());

    try {
        pkt.dump_json_file(out_json_path, 4);
    } catch (const std::exception& ex) {
        std::cerr << "Failed to dump JSON: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
