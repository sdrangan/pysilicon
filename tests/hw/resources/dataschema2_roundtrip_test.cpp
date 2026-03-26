#include <fstream>
#include <iostream>
#include <vector>

#include "demo_packet.h"

int main(int argc, char** argv) {
    const char* in_words_path = (argc > 1) ? argv[1] : "packet_words.txt";
    const char* out_words_path = (argc > 2) ? argv[2] : "packet_words_out.txt";

    std::ifstream in_words(in_words_path);
    if (!in_words) {
        std::cerr << "Failed to open input words file: " << in_words_path << std::endl;
        return 1;
    }

    std::vector<ap_uint<32>> words;
    unsigned long long raw = 0;
    while (in_words >> raw) {
        words.push_back((ap_uint<32>)raw);
    }

    if (words.empty()) {
        std::cerr << "No serialized words found." << std::endl;
        return 1;
    }

    DemoPacket pkt;
    pkt.read_array<32>(words.data());

    constexpr int nwords = __NWORDS__;
    ap_uint<32> out_words[__NWORDS__];
    pkt.write_array<32>(out_words);

    std::ofstream out(out_words_path);
    if (!out) {
        std::cerr << "Failed to open output words file: " << out_words_path << std::endl;
        return 1;
    }

    for (int i = 0; i < nwords; ++i) {
        out << static_cast<unsigned long long>(out_words[i]) << "\n";
    }

    return 0;
}
