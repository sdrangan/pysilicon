#ifndef STREAMUTILS_H
#define STREAMUTILS_H

#include <ap_int.h>
#include <cctype>
#include <cstdlib>
#include <hls_stream.h>
#if __has_include(<hls_axi_stream.h>)
#include <hls_axi_stream.h>
#else
#include <ap_axi_sdata.h>
#endif
#include <stdexcept>
#include <string>

namespace streamutils {

    /**
     * Reinterprets the 32 bits of a float as an unsigned integer
     * without performing any type truncation or rounding.
     */
    inline uint32_t float_to_uint(float f) {
        union {
            float f_val;
            uint32_t u_val;
        } converter;
        converter.f_val = f;
        return converter.u_val;
    }

    /**
     * Reinterprets a 32-bit unsigned integer as a float.
     * Critical for restoring floating point data from a bitstream.
     */
    inline float uint_to_float(uint32_t u) {
        union {
            uint32_t u_val;
            float f_val;
        } converter;
        converter.u_val = u;
        return converter.f_val;
    }

    /**
     * Helper to write a word to an AXI4-Stream with TLAST support.
     * Sets TKEEP and TSTRB to all-ones by default.
     */
    template<int W>
    void write_axi4_word(hls::stream<hls::axis<ap_uint<W>, 0, 0, 0>> &s, ap_uint<W> data, bool tlast) {
        hls::axis<ap_uint<W>, 0, 0, 0> pkt;
        pkt.data = data;
        pkt.last = tlast;
        pkt.keep = -1; // -1 in ap_uint sets all bits to 1
        pkt.strb = -1;
        s.write(pkt);
    }

    inline void json_skip_ws(const std::string& s, size_t& pos) {
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
            ++pos;
        }
    }

    inline void json_expect_char(const std::string& s, size_t& pos, char ch) {
        json_skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != ch) {
            throw std::runtime_error("Malformed JSON: unexpected delimiter.");
        }
        ++pos;
    }

    inline std::string json_parse_string(const std::string& s, size_t& pos) {
        json_skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != '"') {
            throw std::runtime_error("Malformed JSON: expected string key.");
        }
        ++pos;

        std::string out;
        while (pos < s.size()) {
            char c = s[pos++];
            if (c == '"') {
                return out;
            }
            if (c == '\\') {
                if (pos >= s.size()) {
                    throw std::runtime_error("Malformed JSON: invalid escape sequence.");
                }
                char esc = s[pos++];
                switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default:
                    throw std::runtime_error("Malformed JSON: unsupported escape sequence.");
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("Malformed JSON: unterminated string.");
    }

    inline double json_parse_number(const std::string& s, size_t& pos) {
        json_skip_ws(s, pos);
        if (pos >= s.size()) {
            throw std::runtime_error("Malformed JSON: expected number.");
        }

        const char* begin = s.c_str() + pos;
        char* end = nullptr;
        double value = std::strtod(begin, &end);
        if (end == begin) {
            throw std::runtime_error("Malformed JSON: invalid numeric value.");
        }
        pos += static_cast<size_t>(end - begin);
        return value;
    }

} // namespace streamutils

#endif // STREAMUTILS_H