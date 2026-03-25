#ifndef POLY_CMD_HEADER_H
#define POLY_CMD_HEADER_H

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

class PolyCmdHeader {
public:
    ap_uint<16> txn_id;
    CoeffArray coeffs;
    ap_uint<16> nsamp;

    static constexpr int bitwidth = 160;

    static ap_uint<bitwidth> pack_to_uint(const PolyCmdHeader& data) {
        ap_uint<bitwidth> res = 0;
        res.range(15, 0) = data.txn_id;
        res.range(143, 16) = CoeffArray::pack_to_uint(data.coeffs);
        res.range(159, 144) = data.nsamp;
        return res;
    }

    static PolyCmdHeader unpack_from_uint(const ap_uint<bitwidth>& packed) {
        PolyCmdHeader data;
        data.txn_id = (ap_uint<16>)(packed.range(15, 0));
        data.coeffs = CoeffArray::unpack_from_uint(packed.range(143, 16));
        data.nsamp = (ap_uint<16>)(packed.range(159, 144));
        return data;
    }

    template<int word_bw>
    void write_array(ap_uint<word_bw> x[]) const {
        if constexpr (word_bw == 32) {
            x[0] = 0;
            x[0].range(15, 0) = this->txn_id;
            x[1] = streamutils::float_to_uint(this->coeff[0]);
            x[2] = streamutils::float_to_uint(this->coeff[1]);
            x[3] = streamutils::float_to_uint(this->coeff[2]);
            x[4] = streamutils::float_to_uint(this->coeff[3]);
            x[5] = 0;
            x[5].range(15, 0) = this->nsamp;
        }
        else if constexpr (word_bw == 64) {
            x[0] = 0;
            x[0].range(15, 0) = this->txn_id;
            x[1] = 0;
            x[1].range(31, 0) = streamutils::float_to_uint(this->coeff[0]);
            x[1].range(63, 32) = streamutils::float_to_uint(this->coeff[1]);
            x[2] = 0;
            x[2].range(31, 0) = streamutils::float_to_uint(this->coeff[2]);
            x[2].range(63, 32) = streamutils::float_to_uint(this->coeff[3]);
            x[3] = 0;
            x[3].range(15, 0) = this->nsamp;
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for write_array");
        }
    }

    template<int word_bw>
    void write_stream(hls::stream<ap_uint<word_bw>> &s) const {
        if constexpr (word_bw == 32) {
            ap_uint<32> w = 0;
            w.range(15, 0) = this->txn_id;
            s.write(w);
            w = 0;
            w = streamutils::float_to_uint(this->coeff[0]);
            s.write(w);
            w = 0;
            w = streamutils::float_to_uint(this->coeff[1]);
            s.write(w);
            w = 0;
            w = streamutils::float_to_uint(this->coeff[2]);
            s.write(w);
            w = 0;
            w = streamutils::float_to_uint(this->coeff[3]);
            s.write(w);
            w = 0;
            w.range(15, 0) = this->nsamp;
            s.write(w);
        }
        else if constexpr (word_bw == 64) {
            ap_uint<64> w = 0;
            w.range(15, 0) = this->txn_id;
            s.write(w);
            w = 0;
            w.range(31, 0) = streamutils::float_to_uint(this->coeff[0]);
            w.range(63, 32) = streamutils::float_to_uint(this->coeff[1]);
            s.write(w);
            w = 0;
            w.range(31, 0) = streamutils::float_to_uint(this->coeff[2]);
            w.range(63, 32) = streamutils::float_to_uint(this->coeff[3]);
            s.write(w);
            w = 0;
            w.range(15, 0) = this->nsamp;
            s.write(w);
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for write_stream");
        }
    }

    template<int word_bw>
    void write_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s, bool tlast = true) const {
        if constexpr (word_bw == 32) {
            ap_uint<32> w = 0;
            w.range(15, 0) = this->txn_id;
            streamutils::write_axi4_word<32>(s, w, false);
            w = 0;
            w = streamutils::float_to_uint(this->coeff[0]);
            streamutils::write_axi4_word<32>(s, w, false);
            w = 0;
            w = streamutils::float_to_uint(this->coeff[1]);
            streamutils::write_axi4_word<32>(s, w, false);
            w = 0;
            w = streamutils::float_to_uint(this->coeff[2]);
            streamutils::write_axi4_word<32>(s, w, false);
            w = 0;
            w = streamutils::float_to_uint(this->coeff[3]);
            streamutils::write_axi4_word<32>(s, w, false);
            w = 0;
            w.range(15, 0) = this->nsamp;
            streamutils::write_axi4_word<32>(s, w, tlast);
        }
        else if constexpr (word_bw == 64) {
            ap_uint<64> w = 0;
            w.range(15, 0) = this->txn_id;
            streamutils::write_axi4_word<64>(s, w, false);
            w = 0;
            w.range(31, 0) = streamutils::float_to_uint(this->coeff[0]);
            w.range(63, 32) = streamutils::float_to_uint(this->coeff[1]);
            streamutils::write_axi4_word<64>(s, w, false);
            w = 0;
            w.range(31, 0) = streamutils::float_to_uint(this->coeff[2]);
            w.range(63, 32) = streamutils::float_to_uint(this->coeff[3]);
            streamutils::write_axi4_word<64>(s, w, false);
            w = 0;
            w.range(15, 0) = this->nsamp;
            streamutils::write_axi4_word<64>(s, w, tlast);
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for write_axi4_stream");
        }
    }

    template<int word_bw>
    void read_array(const ap_uint<word_bw> x[]) {
        if constexpr (word_bw == 32) {
            this->txn_id = (ap_uint<16>)(x[0].range(15, 0));
            this->coeff[0] = streamutils::uint_to_float((uint32_t)(x[1]));
            this->coeff[1] = streamutils::uint_to_float((uint32_t)(x[2]));
            this->coeff[2] = streamutils::uint_to_float((uint32_t)(x[3]));
            this->coeff[3] = streamutils::uint_to_float((uint32_t)(x[4]));
            this->nsamp = (ap_uint<16>)(x[5].range(15, 0));
        }
        else if constexpr (word_bw == 64) {
            this->txn_id = (ap_uint<16>)(x[0].range(15, 0));
            this->coeff[0] = streamutils::uint_to_float((uint32_t)(x[1].range(31, 0)));
            this->coeff[1] = streamutils::uint_to_float((uint32_t)(x[1].range(63, 32)));
            this->coeff[2] = streamutils::uint_to_float((uint32_t)(x[2].range(31, 0)));
            this->coeff[3] = streamutils::uint_to_float((uint32_t)(x[2].range(63, 32)));
            this->nsamp = (ap_uint<16>)(x[3].range(15, 0));
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for read_array");
        }
    }

    template<int word_bw>
    void read_stream(hls::stream<ap_uint<word_bw>> &s) {
        if constexpr (word_bw == 32) {
            ap_uint<32> w = 0;
            w = s.read();
            this->txn_id = (ap_uint<16>)(w.range(15, 0));
            w = s.read();
            this->coeff[0] = streamutils::uint_to_float((uint32_t)(w));
            w = s.read();
            this->coeff[1] = streamutils::uint_to_float((uint32_t)(w));
            w = s.read();
            this->coeff[2] = streamutils::uint_to_float((uint32_t)(w));
            w = s.read();
            this->coeff[3] = streamutils::uint_to_float((uint32_t)(w));
            w = s.read();
            this->nsamp = (ap_uint<16>)(w.range(15, 0));
        }
        else if constexpr (word_bw == 64) {
            ap_uint<64> w = 0;
            w = s.read();
            this->txn_id = (ap_uint<16>)(w.range(15, 0));
            w = s.read();
            this->coeff[0] = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
            this->coeff[1] = streamutils::uint_to_float((uint32_t)(w.range(63, 32)));
            w = s.read();
            this->coeff[2] = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
            this->coeff[3] = streamutils::uint_to_float((uint32_t)(w.range(63, 32)));
            w = s.read();
            this->nsamp = (ap_uint<16>)(w.range(15, 0));
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for read_stream");
        }
    }

    template<int word_bw>
    void read_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s) {
        if constexpr (word_bw == 32) {
            ap_uint<32> w = 0;
            w = s.read().data;
            this->txn_id = (ap_uint<16>)(w.range(15, 0));
            w = s.read().data;
            this->coeff[0] = streamutils::uint_to_float((uint32_t)(w));
            w = s.read().data;
            this->coeff[1] = streamutils::uint_to_float((uint32_t)(w));
            w = s.read().data;
            this->coeff[2] = streamutils::uint_to_float((uint32_t)(w));
            w = s.read().data;
            this->coeff[3] = streamutils::uint_to_float((uint32_t)(w));
            w = s.read().data;
            this->nsamp = (ap_uint<16>)(w.range(15, 0));
        }
        else if constexpr (word_bw == 64) {
            ap_uint<64> w = 0;
            w = s.read().data;
            this->txn_id = (ap_uint<16>)(w.range(15, 0));
            w = s.read().data;
            this->coeff[0] = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
            this->coeff[1] = streamutils::uint_to_float((uint32_t)(w.range(63, 32)));
            w = s.read().data;
            this->coeff[2] = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
            this->coeff[3] = streamutils::uint_to_float((uint32_t)(w.range(63, 32)));
            w = s.read().data;
            this->nsamp = (ap_uint<16>)(w.range(15, 0));
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for read_axi4_stream");
        }
    }

    void dump_json(std::ostream& os, int indent = 2, int level = 0) const {
        const int step = (indent < 0) ? 0 : indent;
        os << "{";
        os << "\n";
        for (int i = 0; i < (level + 1) * step; ++i) { os << ' '; }
        os << "\"txn_id\": ";
        os << static_cast<unsigned long long>(this->txn_id);
        os << ",";
        os << "\n";
        for (int i = 0; i < (level + 1) * step; ++i) { os << ' '; }
        os << "\"coeffs\": ";
        os << "[";
        for (int i0 = 0; i0 < 4; ++i0) {
        if (i0 > 0) { os << ","; }
        os << this->coeff[i0];
        }
        os << "]";
        os << ",";
        os << "\n";
        for (int i = 0; i < (level + 1) * step; ++i) { os << ' '; }
        os << "\"nsamp\": ";
        os << static_cast<unsigned long long>(this->nsamp);
        os << "\n";
        for (int i = 0; i < (level) * step; ++i) { os << ' '; }
        os << "}";
    }

    void load_json(const std::string& json_text, size_t& pos) {
        streamutils::json_expect_char(json_text, pos, '{');
        bool seen_root_txn_id = false;
        bool seen_root_coeffs = false;
        bool seen_root_nsamp = false;
        bool first = true;
        while (true) {
        streamutils::json_skip_ws(json_text, pos);
        if (pos < json_text.size() && json_text[pos] == '}') {
            ++pos;
            break;
        }
        if (!first) {
            streamutils::json_expect_char(json_text, pos, ',');
        }
        first = false;
        std::string key = streamutils::json_parse_string(json_text, pos);
        streamutils::json_expect_char(json_text, pos, ':');
        if (key == "txn_id") {
            seen_root_txn_id = true;
            this->txn_id = static_cast<ap_uint<16>>(static_cast<unsigned long long>(streamutils::json_parse_number(json_text, pos)));
        }
        else if (key == "coeffs") {
            seen_root_coeffs = true;
            streamutils::json_expect_char(json_text, pos, '[');
            for (int i0 = 0; i0 < 4; ++i0) {
                if (i0 > 0) {
                    streamutils::json_expect_char(json_text, pos, ',');
                }
                this->coeff[i0] = static_cast<float>(streamutils::json_parse_number(json_text, pos));
            }
            streamutils::json_expect_char(json_text, pos, ']');
        }
        else if (key == "nsamp") {
            seen_root_nsamp = true;
            this->nsamp = static_cast<ap_uint<16>>(static_cast<unsigned long long>(streamutils::json_parse_number(json_text, pos)));
        }
        else {
            throw std::runtime_error("Malformed JSON: unexpected key for schema.");
        }
        }
        if (!seen_root_txn_id) {
        throw std::runtime_error("Malformed JSON: missing required key 'txn_id'.");
        }
        if (!seen_root_coeffs) {
        throw std::runtime_error("Malformed JSON: missing required key 'coeffs'.");
        }
        if (!seen_root_nsamp) {
        throw std::runtime_error("Malformed JSON: missing required key 'nsamp'.");
        }
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

#endif // POLY_CMD_HEADER_H