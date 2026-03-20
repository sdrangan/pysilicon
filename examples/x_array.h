#ifndef X_ARRAY_H
#define X_ARRAY_H

#include <ap_int.h>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <hls_stream.h>
#include <iterator>
#include <stdexcept>
#include <string>
#if __has_include(<hls_axi_stream.h>)
#include <hls_axi_stream.h>
#else
#include <ap_axi_sdata.h>
#endif
#include <iostream>
#include "streamutils.h"

class XArray {
public:
    float x_elem[1024];

    static constexpr int bitwidth = 32768;

    static ap_uint<bitwidth> pack_to_uint(const XArray& data) {
        ap_uint<bitwidth> res = 0;
        int bitpos = 0;
        for (int i0 = 0; i0 < 1024; ++i0) {
            res.range(bitpos + 32 - 1, bitpos) = streamutils::float_to_uint(data.x_elem[i0]);
            bitpos += 32;
        }
        return res;
    }

    static XArray unpack_from_uint(const ap_uint<bitwidth>& packed) {
        XArray data;
        int bitpos = 0;
        for (int i0 = 0; i0 < 1024; ++i0) {
            data.x_elem[i0] = streamutils::uint_to_float((uint32_t)(packed.range(bitpos + 32 - 1, bitpos)));
            bitpos += 32;
        }
        return data;
    }

    template<int word_bw>
    static constexpr int pf() {
        return word_bw / 32;
    }

    template<int word_bw>
    static void write_word(const float in[pf<word_bw>()], ap_uint<word_bw>& w, int n = pf<word_bw>()) {
        #pragma HLS INLINE
        if constexpr (word_bw == 32) {
            w = 0;
            if (n > 0) {
                w.range(31, 0) = streamutils::float_to_uint(in[0]);
            }
        }
        else if constexpr (word_bw == 64) {
            w = 0;
            if (n > 0) {
                w.range(31, 0) = streamutils::float_to_uint(in[0]);
            }
            if (n > 1) {
                w.range(63, 32) = streamutils::float_to_uint(in[1]);
            }
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for write_word");
        }
    }

    template<int word_bw>
    static void read_word(float out[pf<word_bw>()], const ap_uint<word_bw>& w, int n = pf<word_bw>()) {
        #pragma HLS INLINE
        if constexpr (word_bw == 32) {
            if (n > 0) {
                out[0] = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
            }
        }
        else if constexpr (word_bw == 64) {
            if (n > 0) {
                out[0] = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
            }
            if (n > 1) {
                out[1] = streamutils::uint_to_float((uint32_t)(w.range(63, 32)));
            }
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for read_word");
        }
    }

    template<int word_bw>
    void write_array(ap_uint<word_bw> x[], int n0=1) const {
        if constexpr (word_bw == 32) {
            const int n0_eff = (n0 < 0) ? 0 : ((n0 > 1024) ? 1024 : n0);
            int out_idx = 0;
            for (int i0 = 0; i0 < n0_eff; ++i0) {
                x[out_idx++] = streamutils::float_to_uint(this->x_elem[i0]);
            }
        }
        else if constexpr (word_bw == 64) {
            const int n0_eff = (n0 < 0) ? 0 : ((n0 > 1024) ? 1024 : n0);
            int out_idx = 0;
            for (int i = 0; i < n0_eff; i += 2) {
                #pragma HLS PIPELINE II=1
                ap_uint<64> w = 0;
                if (i + 0 < n0_eff) {
                    w.range(31, 0) = streamutils::float_to_uint(this->x_elem[i + 0]);
                }
                if (i + 1 < n0_eff) {
                    w.range(63, 32) = streamutils::float_to_uint(this->x_elem[i + 1]);
                }
                x[out_idx++] = w;
            }
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for write_array");
        }
    }

    template<int word_bw>
    void write_stream(hls::stream<ap_uint<word_bw>> &s, int n0=1) const {
        if constexpr (word_bw == 32) {
            ap_uint<32> w = 0;
            const int n0_eff = (n0 < 0) ? 0 : ((n0 > 1024) ? 1024 : n0);
            int out_idx = 0;
            ap_uint<32> w = 0;
            for (int i0 = 0; i0 < n0_eff; ++i0) {
                w = streamutils::float_to_uint(this->x_elem[i0]);
                s.write(w);
                out_idx++;
            }
        }
        else if constexpr (word_bw == 64) {
            ap_uint<64> w = 0;
            const int n0_eff = (n0 < 0) ? 0 : ((n0 > 1024) ? 1024 : n0);
            int out_idx = 0;
            for (int i = 0; i < n0_eff; i += 2) {
                #pragma HLS PIPELINE II=1
                ap_uint<64> w = 0;
                if (i + 0 < n0_eff) {
                    w.range(31, 0) = streamutils::float_to_uint(this->x_elem[i + 0]);
                }
                if (i + 1 < n0_eff) {
                    w.range(63, 32) = streamutils::float_to_uint(this->x_elem[i + 1]);
                }
                s.write(w);
                out_idx++;
            }
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for write_stream");
        }
    }

    template<int word_bw>
    void write_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s, bool tlast = true, int n0=1) const {
        if constexpr (word_bw == 32) {
            ap_uint<32> w = 0;
            const int n0_eff = (n0 < 0) ? 0 : ((n0 > 1024) ? 1024 : n0);
            int out_idx = 0;
            ap_uint<32> w = 0;
            for (int i0 = 0; i0 < n0_eff; ++i0) {
                w = streamutils::float_to_uint(this->x_elem[i0]);
                const bool last = (out_idx == (n0_eff) - 1) ? tlast : false;
                streamutils::write_axi4_word<32>(s, w, last);
                out_idx++;
            }
        }
        else if constexpr (word_bw == 64) {
            ap_uint<64> w = 0;
            const int n0_eff = (n0 < 0) ? 0 : ((n0 > 1024) ? 1024 : n0);
            int out_idx = 0;
            const int total_words = (n0_eff + 2 - 1) / 2;
            for (int i = 0; i < n0_eff; i += 2) {
                #pragma HLS PIPELINE II=1
                ap_uint<64> w = 0;
                if (i + 0 < n0_eff) {
                    w.range(31, 0) = streamutils::float_to_uint(this->x_elem[i + 0]);
                }
                if (i + 1 < n0_eff) {
                    w.range(63, 32) = streamutils::float_to_uint(this->x_elem[i + 1]);
                }
                const bool last = (out_idx == total_words - 1) ? tlast : false;
                streamutils::write_axi4_word<64>(s, w, last);
                out_idx++;
            }
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for write_axi4_stream");
        }
    }

    template<int word_bw>
    void read_array(const ap_uint<word_bw> x[], int n0=1) {
        if constexpr (word_bw == 32) {
            const int n0_eff = (n0 < 0) ? 0 : ((n0 > 1024) ? 1024 : n0);
            int in_idx = 0;
            for (int i0 = 0; i0 < n0_eff; ++i0) {
                this->x_elem[i0] = streamutils::uint_to_float((uint32_t)(x[in_idx]));
                in_idx++;
            }
        }
        else if constexpr (word_bw == 64) {
            const int n0_eff = (n0 < 0) ? 0 : ((n0 > 1024) ? 1024 : n0);
            int in_idx = 0;
            for (int i = 0; i < n0_eff; i += 2) {
                #pragma HLS PIPELINE II=1
                ap_uint<64> w = x[in_idx++];
                if (i + 0 < n0_eff) {
                    this->x_elem[i + 0] = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
                }
                if (i + 1 < n0_eff) {
                    this->x_elem[i + 1] = streamutils::uint_to_float((uint32_t)(w.range(63, 32)));
                }
            }
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for read_array");
        }
    }

    template<int word_bw>
    void read_stream(hls::stream<ap_uint<word_bw>> &s, int n0=1) {
        if constexpr (word_bw == 32) {
            ap_uint<32> w = 0;
            const int n0_eff = (n0 < 0) ? 0 : ((n0 > 1024) ? 1024 : n0);
            int in_idx = 0;
            ap_uint<32> w = 0;
            for (int i0 = 0; i0 < n0_eff; ++i0) {
                w = s.read();
                this->x_elem[i0] = streamutils::uint_to_float((uint32_t)(w));
                in_idx++;
            }
        }
        else if constexpr (word_bw == 64) {
            ap_uint<64> w = 0;
            const int n0_eff = (n0 < 0) ? 0 : ((n0 > 1024) ? 1024 : n0);
            int in_idx = 0;
            for (int i = 0; i < n0_eff; i += 2) {
                #pragma HLS PIPELINE II=1
                ap_uint<64> w = s.read();
                in_idx++;
                if (i + 0 < n0_eff) {
                    this->x_elem[i + 0] = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
                }
                if (i + 1 < n0_eff) {
                    this->x_elem[i + 1] = streamutils::uint_to_float((uint32_t)(w.range(63, 32)));
                }
            }
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for read_stream");
        }
    }

    template<int word_bw>
    void read_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s, int n0=1) {
        if constexpr (word_bw == 32) {
            ap_uint<32> w = 0;
            const int n0_eff = (n0 < 0) ? 0 : ((n0 > 1024) ? 1024 : n0);
            int in_idx = 0;
            ap_uint<32> w = 0;
            for (int i0 = 0; i0 < n0_eff; ++i0) {
                w = s.read().data;
                this->x_elem[i0] = streamutils::uint_to_float((uint32_t)(w));
                in_idx++;
            }
        }
        else if constexpr (word_bw == 64) {
            ap_uint<64> w = 0;
            const int n0_eff = (n0 < 0) ? 0 : ((n0 > 1024) ? 1024 : n0);
            int in_idx = 0;
            for (int i = 0; i < n0_eff; i += 2) {
                #pragma HLS PIPELINE II=1
                ap_uint<64> w = s.read().data;
                in_idx++;
                if (i + 0 < n0_eff) {
                    this->x_elem[i + 0] = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
                }
                if (i + 1 < n0_eff) {
                    this->x_elem[i + 1] = streamutils::uint_to_float((uint32_t)(w.range(63, 32)));
                }
            }
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for read_axi4_stream");
        }
    }

    void dump_json(std::ostream& os, int indent = 2, int level = 0) const {
        const int step = (indent < 0) ? 0 : indent;
        os << "[";
        for (int i0 = 0; i0 < 1024; ++i0) {
        if (i0 > 0) { os << ","; }
        os << this->x_elem[i0];
        }
        os << "]";
    }

    void load_json(const std::string& json_text, size_t& pos) {
        streamutils::json_expect_char(json_text, pos, '[');
        for (int i0 = 0; i0 < 1024; ++i0) {
        if (i0 > 0) {
            streamutils::json_expect_char(json_text, pos, ',');
        }
        this->x_elem[i0] = static_cast<float>(streamutils::json_parse_number(json_text, pos));
        }
        streamutils::json_expect_char(json_text, pos, ']');
    }

    void load_json(std::istream& is) {
        std::string json_text((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
        size_t pos = 0;
        streamutils::json_skip_ws(json_text, pos);
        this->load_json(json_text, pos);
        streamutils::json_skip_ws(json_text, pos);
        if (pos != json_text.size()) {
            throw std::runtime_error("Trailing characters after JSON object.");
        }
    }

#ifndef __SYNTHESIS__
    void dump_json_file(const char* file_path, int indent = 2) const {
        std::ofstream ofs(file_path);
        if (!ofs) {
            throw std::runtime_error("Failed to open output JSON file.");
        }
        this->dump_json(ofs, indent);
    }

    void load_json_file(const char* file_path) {
        std::ifstream ifs(file_path);
        if (!ifs) {
            throw std::runtime_error("Failed to open input JSON file.");
        }
        this->load_json(ifs);
    }
#endif

};

#endif // X_ARRAY_H