__EXTRA_INCLUDES__
#include "__HEADER__"

#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const char* in_path  = (argc > 1) ? argv[1] : "dataunion_words_in.txt";
    const char* out_path = (argc > 2) ? argv[2] : "dataunion_words_out.txt";

    static constexpr int NWORDS   = __NWORDS__;
    static constexpr int NPACKETS = __NPACKETS__;
    static constexpr int WORD_BW  = __WORD_BW__;

    std::ifstream fin(in_path);
    if (!fin) {
        std::cerr << "Cannot open input file: " << in_path << std::endl;
        return 1;
    }

    std::vector<ap_uint<WORD_BW>> words_all;
    unsigned long long v;
    while (fin >> v) words_all.push_back((ap_uint<WORD_BW>)v);

    if ((int)words_all.size() != NWORDS * NPACKETS) {
        std::cerr << "Expected " << (NWORDS * NPACKETS)
                  << " words, got " << words_all.size() << std::endl;
        return 1;
    }

    std::ofstream fout(out_path);
    if (!fout) {
        std::cerr << "Cannot open output file: " << out_path << std::endl;
        return 1;
    }

    for (int p = 0; p < NPACKETS; p++) {
        ap_uint<WORD_BW> words_in[NWORDS];
        for (int i = 0; i < NWORDS; i++) words_in[i] = words_all[p * NWORDS + i];

        __DATAUNION_CLASS__ du;
        du.read_array<WORD_BW>(words_in);

        ap_uint<WORD_BW> words_out[NWORDS];
        du.write_array<WORD_BW>(words_out);

        for (int i = 0; i < NWORDS; i++) {
            fout << (unsigned long long)((ap_uint<64>)words_out[i]) << "\n";
        }
    }

    return 0;
}
