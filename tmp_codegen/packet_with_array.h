#ifndef PACKET_WITH_ARRAY_H
#define PACKET_WITH_ARRAY_H

#include <ap_int.h>
#include <hls_stream.h>
#if __has_include(<hls_axi_stream.h>)
#include <hls_axi_stream.h>
#else
#include <ap_axi_sdata.h>
#endif
#include "streamutils_hls.h"

#include "float32_array.h"

struct PacketWithArray {
    ap_uint<16> count;
    Float32Array coeffs;

    static constexpr int bitwidth = 144;

    template<int word_bw>
    struct nwords_impl {
        static constexpr int value() {
            static_assert(word_bw < 0, "Unsupported word_bw for nwords");
            return 0;
        }
    };

    template<>
    struct nwords_impl<32> {
        static constexpr int value() {
            return 5;
        }
    };

    template<>
    struct nwords_impl<64> {
        static constexpr int value() {
            return 3;
        }
    };

    template<int word_bw>
    static constexpr int nwords() {
        return nwords_impl<word_bw>::value();
    }

    static ap_uint<bitwidth> pack_to_uint(const PacketWithArray& data) {
        ap_uint<bitwidth> res = 0;
        res.range(15, 0) = data.count;
        res.range(143, 16) = Float32Array::pack_to_uint(data.coeffs);
        return res;
    }

    static PacketWithArray unpack_from_uint(const ap_uint<bitwidth>& packed) {
        PacketWithArray data;
        data.count = (ap_uint<16>)(packed.range(15, 0));
        data.coeffs = Float32Array::unpack_from_uint(packed.range(143, 16));
        return data;
    }

    template<int word_bw>
    struct write_array_impl {
        static void run(const PacketWithArray* self, ap_uint<word_bw> x[]) {
            static_assert(word_bw < 0, "Unsupported word_bw for write_array");
            (void)self;
            (void)x;
        }
    };

    template<>
    struct write_array_impl<32> {
        static void run(const PacketWithArray* self, ap_uint<32> x[]) {
            x[0] = 0;
            x[0].range(15, 0) = self->count;
            const int n0_eff = 4;
            int out_idx = 1;
            for (int i0 = 0; i0 < n0_eff; ++i0) {
                x[out_idx++] = streamutils::float_to_uint(self->coeffs.coeff[i0]);
            }
        }
    };

    template<>
    struct write_array_impl<64> {
        static void run(const PacketWithArray* self, ap_uint<64> x[]) {
            x[0] = 0;
            x[0].range(15, 0) = self->count;
            const int n0_eff = 4;
            int out_idx = 1;
            for (int i = 0; i < n0_eff; i += 2) {
                #pragma HLS PIPELINE II=1
                ap_uint<64> w = 0;
                if (i + 0 < n0_eff) {
                    w.range(31, 0) = streamutils::float_to_uint(self->coeffs.coeff[i + 0]);
                }
                if (i + 1 < n0_eff) {
                    w.range(63, 32) = streamutils::float_to_uint(self->coeffs.coeff[i + 1]);
                }
                x[out_idx++] = w;
            }
        }
    };

    template<int word_bw>
    void write_array(ap_uint<word_bw> x[]) const {
        write_array_impl<word_bw>::run(this, x);
    }

    template<int word_bw>
    struct write_stream_impl {
        static void run(const PacketWithArray* self, hls::stream<ap_uint<word_bw>> &s) {
            static_assert(word_bw < 0, "Unsupported word_bw for write_stream");
            (void)self;
            (void)s;
        }
    };

    template<>
    struct write_stream_impl<32> {
        static void run(const PacketWithArray* self, hls::stream<ap_uint<32>> &s) {
            ap_uint<32> w = 0;
            w.range(15, 0) = self->count;
            s.write(w);
            w = 0;
            const int n0_eff = 4;
            int out_idx = 0;
            for (int i0 = 0; i0 < n0_eff; ++i0) {
                w = streamutils::float_to_uint(self->coeffs.coeff[i0]);
                s.write(w);
                out_idx++;
            }
        }
    };

    template<>
    struct write_stream_impl<64> {
        static void run(const PacketWithArray* self, hls::stream<ap_uint<64>> &s) {
            ap_uint<64> w = 0;
            w.range(15, 0) = self->count;
            s.write(w);
            w = 0;
            const int n0_eff = 4;
            int out_idx = 0;
            for (int i = 0; i < n0_eff; i += 2) {
                #pragma HLS PIPELINE II=1
                w = 0;
                if (i + 0 < n0_eff) {
                    w.range(31, 0) = streamutils::float_to_uint(self->coeffs.coeff[i + 0]);
                }
                if (i + 1 < n0_eff) {
                    w.range(63, 32) = streamutils::float_to_uint(self->coeffs.coeff[i + 1]);
                }
                s.write(w);
                out_idx++;
            }
        }
    };

    template<int word_bw>
    void write_stream(hls::stream<ap_uint<word_bw>> &s) const {
        write_stream_impl<word_bw>::run(this, s);
    }

    template<int word_bw>
    struct write_axi4_stream_impl {
        static void run(const PacketWithArray* self, hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s, bool tlast) {
            static_assert(word_bw < 0, "Unsupported word_bw for write_axi4_stream");
            (void)self;
            (void)s;
            (void)tlast;
        }
    };

    template<>
    struct write_axi4_stream_impl<32> {
        static void run(const PacketWithArray* self, hls::stream<hls::axis<ap_uint<32>, 0, 0, 0>> &s, bool tlast) {
            ap_uint<32> w = 0;
            w.range(15, 0) = self->count;
            streamutils::write_axi4_word<32>(s, w, false);
            w = 0;
            const int n0_eff = 4;
            int out_idx = 0;
            for (int i0 = 0; i0 < n0_eff; ++i0) {
                w = streamutils::float_to_uint(self->coeffs.coeff[i0]);
                const bool last = (out_idx == (n0_eff) - 1) ? tlast : false;
                streamutils::write_axi4_word<32>(s, w, last);
                out_idx++;
            }
        }
    };

    template<>
    struct write_axi4_stream_impl<64> {
        static void run(const PacketWithArray* self, hls::stream<hls::axis<ap_uint<64>, 0, 0, 0>> &s, bool tlast) {
            ap_uint<64> w = 0;
            w.range(15, 0) = self->count;
            streamutils::write_axi4_word<64>(s, w, false);
            w = 0;
            const int n0_eff = 4;
            int out_idx = 0;
            const int total_words = (n0_eff + 2 - 1) / 2;
            for (int i = 0; i < n0_eff; i += 2) {
                #pragma HLS PIPELINE II=1
                w = 0;
                if (i + 0 < n0_eff) {
                    w.range(31, 0) = streamutils::float_to_uint(self->coeffs.coeff[i + 0]);
                }
                if (i + 1 < n0_eff) {
                    w.range(63, 32) = streamutils::float_to_uint(self->coeffs.coeff[i + 1]);
                }
                const bool last = (out_idx == total_words - 1) ? tlast : false;
                streamutils::write_axi4_word<64>(s, w, last);
                out_idx++;
            }
        }
    };

    template<int word_bw>
    void write_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s, bool tlast = true) const {
        write_axi4_stream_impl<word_bw>::run(this, s, tlast);
    }

    template<int word_bw>
    struct read_array_impl {
        static void run(PacketWithArray* self, const ap_uint<word_bw> x[]) {
            static_assert(word_bw < 0, "Unsupported word_bw for read_array");
            (void)self;
            (void)x;
        }
    };

    template<>
    struct read_array_impl<32> {
        static void run(PacketWithArray* self, const ap_uint<32> x[]) {
            self->count = (ap_uint<16>)(x[0].range(15, 0));
            const int n0_eff = 4;
            int in_idx = 1;
            for (int i0 = 0; i0 < n0_eff; ++i0) {
                self->coeffs.coeff[i0] = streamutils::uint_to_float((uint32_t)(x[in_idx]));
                in_idx++;
            }
        }
    };

    template<>
    struct read_array_impl<64> {
        static void run(PacketWithArray* self, const ap_uint<64> x[]) {
            self->count = (ap_uint<16>)(x[0].range(15, 0));
            const int n0_eff = 4;
            int in_idx = 1;
            for (int i = 0; i < n0_eff; i += 2) {
                #pragma HLS PIPELINE II=1
                ap_uint<64> w = x[in_idx++];
                if (i + 0 < n0_eff) {
                    self->coeffs.coeff[i + 0] = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
                }
                if (i + 1 < n0_eff) {
                    self->coeffs.coeff[i + 1] = streamutils::uint_to_float((uint32_t)(w.range(63, 32)));
                }
            }
        }
    };

    template<int word_bw>
    void read_array(const ap_uint<word_bw> x[]) {
        read_array_impl<word_bw>::run(this, x);
    }

    template<int word_bw>
    struct read_stream_impl {
        static void run(PacketWithArray* self, hls::stream<ap_uint<word_bw>> &s) {
            static_assert(word_bw < 0, "Unsupported word_bw for read_stream");
            (void)self;
            (void)s;
        }
    };

    template<>
    struct read_stream_impl<32> {
        static void run(PacketWithArray* self, hls::stream<ap_uint<32>> &s) {
            ap_uint<32> w = 0;
            w = s.read();
            self->count = (ap_uint<16>)(w.range(15, 0));
            const int n0_eff = 4;
            int in_idx = 1;
            for (int i0 = 0; i0 < n0_eff; ++i0) {
                w = s.read();
                self->coeffs.coeff[i0] = streamutils::uint_to_float((uint32_t)(w));
                in_idx++;
            }
        }
    };

    template<>
    struct read_stream_impl<64> {
        static void run(PacketWithArray* self, hls::stream<ap_uint<64>> &s) {
            ap_uint<64> w = 0;
            w = s.read();
            self->count = (ap_uint<16>)(w.range(15, 0));
            const int n0_eff = 4;
            int in_idx = 1;
            for (int i = 0; i < n0_eff; i += 2) {
                #pragma HLS PIPELINE II=1
                w = s.read();
                in_idx++;
                if (i + 0 < n0_eff) {
                    self->coeffs.coeff[i + 0] = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
                }
                if (i + 1 < n0_eff) {
                    self->coeffs.coeff[i + 1] = streamutils::uint_to_float((uint32_t)(w.range(63, 32)));
                }
            }
        }
    };

    template<int word_bw>
    void read_stream(hls::stream<ap_uint<word_bw>> &s) {
        read_stream_impl<word_bw>::run(this, s);
    }

    template<int word_bw>
    struct read_axi4_stream_impl {
        static void run(PacketWithArray* self, hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s) {
            static_assert(word_bw < 0, "Unsupported word_bw for read_axi4_stream");
            (void)self;
            (void)s;
        }
    };

    template<>
    struct read_axi4_stream_impl<32> {
        static void run(PacketWithArray* self, hls::stream<hls::axis<ap_uint<32>, 0, 0, 0>> &s) {
            ap_uint<32> w = 0;
            w = s.read().data;
            self->count = (ap_uint<16>)(w.range(15, 0));
            const int n0_eff = 4;
            int in_idx = 1;
            for (int i0 = 0; i0 < n0_eff; ++i0) {
                w = s.read().data;
                self->coeffs.coeff[i0] = streamutils::uint_to_float((uint32_t)(w));
                in_idx++;
            }
        }
    };

    template<>
    struct read_axi4_stream_impl<64> {
        static void run(PacketWithArray* self, hls::stream<hls::axis<ap_uint<64>, 0, 0, 0>> &s) {
            ap_uint<64> w = 0;
            w = s.read().data;
            self->count = (ap_uint<16>)(w.range(15, 0));
            const int n0_eff = 4;
            int in_idx = 1;
            for (int i = 0; i < n0_eff; i += 2) {
                #pragma HLS PIPELINE II=1
                w = s.read().data;
                in_idx++;
                if (i + 0 < n0_eff) {
                    self->coeffs.coeff[i + 0] = streamutils::uint_to_float((uint32_t)(w.range(31, 0)));
                }
                if (i + 1 < n0_eff) {
                    self->coeffs.coeff[i + 1] = streamutils::uint_to_float((uint32_t)(w.range(63, 32)));
                }
            }
        }
    };

    template<int word_bw>
    void read_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s) {
        read_axi4_stream_impl<word_bw>::run(this, s);
    }

#ifdef PYSILICON_ENABLE_PACKET_WITH_ARRAY_TB_H_MEMBERS
    void dump_json(std::ostream& os, int indent = 2, int level = 0) const;
    void load_json(const std::string& json_text, size_t& pos);
    void load_json(std::istream& is);
    void dump_json_file(const char* file_path, int indent = 2) const;
    void load_json_file(const char* file_path);
#endif
};

#endif // PACKET_WITH_ARRAY_H