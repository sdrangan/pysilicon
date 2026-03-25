#ifndef POLY_RESP_FOOTER_H
#define POLY_RESP_FOOTER_H

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

class PolyRespFooter {
public:
    ap_uint<16> ndata_read;
    PolyError err_code;

    static constexpr int bitwidth = 24;

    template<int word_bw>
    static constexpr int nwords() {
        if constexpr (word_bw == 32) {
            return 1;
        }
        else if constexpr (word_bw == 64) {
            return 1;
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for nwords");
            return 0;
        }
    }

    static ap_uint<bitwidth> pack_to_uint(const PolyRespFooter& data) {
        ap_uint<bitwidth> res = 0;
        res.range(15, 0) = data.ndata_read;
        res.range(23, 16) = (ap_uint<8>)(data.err_code);
        return res;
    }

    static PolyRespFooter unpack_from_uint(const ap_uint<bitwidth>& packed) {
        PolyRespFooter data;
        data.ndata_read = (ap_uint<16>)(packed.range(15, 0));
        data.err_code = (PolyError)(packed.range(23, 16));
        return data;
    }

    template<int word_bw>
    void write_array(ap_uint<word_bw> x[]) const {
        if constexpr (word_bw == 32) {
            x[0] = 0;
            x[0].range(15, 0) = this->ndata_read;
            x[0].range(23, 16) = (ap_uint<8>)(this->err_code);
        }
        else if constexpr (word_bw == 64) {
            x[0] = 0;
            x[0].range(15, 0) = this->ndata_read;
            x[0].range(23, 16) = (ap_uint<8>)(this->err_code);
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for write_array");
        }
    }

    template<int word_bw>
    void write_stream(hls::stream<ap_uint<word_bw>> &s) const {
        if constexpr (word_bw == 32) {
            ap_uint<32> w = 0;
            w.range(15, 0) = this->ndata_read;
            w.range(23, 16) = (ap_uint<8>)(this->err_code);
            s.write(w);
        }
        else if constexpr (word_bw == 64) {
            ap_uint<64> w = 0;
            w.range(15, 0) = this->ndata_read;
            w.range(23, 16) = (ap_uint<8>)(this->err_code);
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
            w.range(15, 0) = this->ndata_read;
            w.range(23, 16) = (ap_uint<8>)(this->err_code);
            streamutils::write_axi4_word<32>(s, w, tlast);
        }
        else if constexpr (word_bw == 64) {
            ap_uint<64> w = 0;
            w.range(15, 0) = this->ndata_read;
            w.range(23, 16) = (ap_uint<8>)(this->err_code);
            streamutils::write_axi4_word<64>(s, w, tlast);
        }
        else {
            static_assert(word_bw > 0, "Unsupported word_bw for write_axi4_stream");
        }
    }

    template<int word_bw>
    void read_array(const ap_uint<word_bw> x[]) {
        if constexpr (word_bw == 32) {
            this->ndata_read = (ap_uint<16>)(x[0].range(15, 0));
            this->err_code = (PolyError)(x[0].range(23, 16));
        }
        else if constexpr (word_bw == 64) {
            this->ndata_read = (ap_uint<16>)(x[0].range(15, 0));
            this->err_code = (PolyError)(x[0].range(23, 16));
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
            this->ndata_read = (ap_uint<16>)(w.range(15, 0));
            this->err_code = (PolyError)(w.range(23, 16));
        }
        else if constexpr (word_bw == 64) {
            ap_uint<64> w = 0;
            w = s.read();
            this->ndata_read = (ap_uint<16>)(w.range(15, 0));
            this->err_code = (PolyError)(w.range(23, 16));
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
            this->ndata_read = (ap_uint<16>)(w.range(15, 0));
            this->err_code = (PolyError)(w.range(23, 16));
        }
        else if constexpr (word_bw == 64) {
            ap_uint<64> w = 0;
            w = s.read().data;
            this->ndata_read = (ap_uint<16>)(w.range(15, 0));
            this->err_code = (PolyError)(w.range(23, 16));
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
        os << "\"ndata_read\": ";
        os << static_cast<unsigned long long>(this->ndata_read);
        os << ",";
        os << "\n";
        for (int i = 0; i < (level + 1) * step; ++i) { os << ' '; }
        os << "\"err_code\": ";
        os << this->err_code;
        os << "\n";
        for (int i = 0; i < (level) * step; ++i) { os << ' '; }
        os << "}";
    }

    void load_json(const std::string& json_text, size_t& pos) {
        streamutils::json_expect_char(json_text, pos, '{');
        bool seen_root_ndata_read = false;
        bool seen_root_err_code = false;
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
        if (key == "ndata_read") {
            seen_root_ndata_read = true;
            this->ndata_read = static_cast<ap_uint<16>>(static_cast<unsigned long long>(streamutils::json_parse_number(json_text, pos)));
        }
        else if (key == "err_code") {
            seen_root_err_code = true;
            this->err_code = static_cast<PolyError>(static_cast<long long>(streamutils::json_parse_number(json_text, pos)));
        }
        else {
            throw std::runtime_error("Malformed JSON: unexpected key for schema.");
        }
        }
        if (!seen_root_ndata_read) {
        throw std::runtime_error("Malformed JSON: missing required key 'ndata_read'.");
        }
        if (!seen_root_err_code) {
        throw std::runtime_error("Malformed JSON: missing required key 'err_code'.");
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

#endif // POLY_RESP_FOOTER_H