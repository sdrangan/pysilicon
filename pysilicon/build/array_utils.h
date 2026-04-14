#ifndef PYSILICON_CODEGEN_ARRAY_UTILS_H
#define PYSILICON_CODEGEN_ARRAY_UTILS_H

#include <ap_int.h>

#include <cstdint>
#include <type_traits>

namespace array_utils {

namespace detail {

template <typename...>
using void_t = void;

template <int word_bw>
struct ReadCursor {
    int word_idx;
    int bit_idx;

    ReadCursor() : word_idx(0), bit_idx(0) {}
};

inline float uint32_to_float(uint32_t value) {
    union {
        uint32_t u32;
        float f32;
    } converter;
    converter.u32 = value;
    return converter.f32;
}

template <typename T>
struct is_ap_uint : std::false_type {};

template <int width>
struct is_ap_uint<ap_uint<width>> : std::true_type {};

template <typename T>
struct is_ap_int : std::false_type {};

template <int width>
struct is_ap_int<ap_int<width>> : std::true_type {};

template <typename T, typename Enable = void>
struct bit_width;

template <typename T>
struct bit_width<T, typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value>::type> {
    static constexpr int value = static_cast<int>(sizeof(T) * 8);
};

template <>
struct bit_width<float> {
    static constexpr int value = 32;
};

template <int width>
struct bit_width<ap_uint<width>> {
    static constexpr int value = width;
};

template <int width>
struct bit_width<ap_int<width>> {
    static constexpr int value = width;
};

template <typename T, typename Enable = void>
struct is_supported_scalar : std::false_type {};

template <typename T>
struct is_supported_scalar<T, typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value>::type>
    : std::true_type {};

template <>
struct is_supported_scalar<float> : std::true_type {};

template <int width>
struct is_supported_scalar<ap_uint<width>> : std::true_type {};

template <int width>
struct is_supported_scalar<ap_int<width>> : std::true_type {};

template <typename T, typename = void>
struct has_bitwidth : std::false_type {};

template <typename T>
struct has_bitwidth<T, void_t<decltype(T::bitwidth)>> : std::true_type {};

template <typename T, typename = void>
struct has_unpack_from_uint : std::false_type {};

template <typename T>
struct has_unpack_from_uint<T, void_t<decltype(T::unpack_from_uint(std::declval<ap_uint<T::bitwidth>>()))>>
    : std::true_type {};

template <typename T, int word_bw, typename = void>
struct has_read_recursive_inplace : std::false_type {};

template <typename T, int word_bw>
struct has_read_recursive_inplace<
    T,
    word_bw,
    void_t<decltype(T::template read_recursive<word_bw>(
        std::declval<const ap_uint<word_bw>*>(),
        std::declval<int&>(),
        std::declval<int&>(),
        std::declval<T&>()))>> : std::true_type {};

template <typename T, int word_bw, typename = void>
struct has_read_recursive_return : std::false_type {};

template <typename T, int word_bw>
struct has_read_recursive_return<
    T,
    word_bw,
    void_t<decltype(T::template read_recursive<word_bw>(
        std::declval<const ap_uint<word_bw>*>(),
        std::declval<int&>(),
        std::declval<int&>()))>> {
    static constexpr bool value = std::is_same<
        decltype(T::template read_recursive<word_bw>(
            std::declval<const ap_uint<word_bw>*>(),
            std::declval<int&>(),
            std::declval<int&>())),
        T>::value;
};

template <typename T, typename = void>
struct has_schema_unpack_api : std::false_type {};

template <typename T>
struct has_schema_unpack_api<T, typename std::enable_if<has_bitwidth<T>::value && has_unpack_from_uint<T>::value>::type>
    : std::true_type {};

template <int width, int word_bw>
ap_uint<width> read_packed_bits(const ap_uint<word_bw>* src, ReadCursor<word_bw>& cursor) {
    static_assert(width > 0, "Packed element width must be positive.");
    ap_uint<width> bits = 0;

    if constexpr (width <= word_bw) {
        if (cursor.bit_idx + width > word_bw) {
            ++cursor.word_idx;
            cursor.bit_idx = 0;
        }

        const int hi = cursor.bit_idx + width - 1;
        const int lo = cursor.bit_idx;
        bits = src[cursor.word_idx].range(hi, lo);

        cursor.bit_idx += width;
        if (cursor.bit_idx == word_bw) {
            ++cursor.word_idx;
            cursor.bit_idx = 0;
        }
    }
    else {
        if (cursor.bit_idx != 0) {
            ++cursor.word_idx;
            cursor.bit_idx = 0;
        }

        int dst_bit = 0;
        while (dst_bit < width) {
            constexpr int take_full = word_bw;
            const int remaining = width - dst_bit;
            const int take = (remaining < take_full) ? remaining : take_full;
            bits.range(dst_bit + take - 1, dst_bit) = src[cursor.word_idx].range(take - 1, 0);
            ++cursor.word_idx;
            dst_bit += take;
        }
    }

    return bits;
}

template <typename T, int width>
typename std::enable_if<std::is_same<T, float>::value, T>::type scalar_from_bits(const ap_uint<width>& bits) {
    static_assert(width == 32, "float deserialization expects 32 packed bits.");
    return uint32_to_float(static_cast<uint32_t>(bits));
}

template <typename T, int width>
typename std::enable_if<is_ap_uint<T>::value, T>::type scalar_from_bits(const ap_uint<width>& bits) {
    return T(bits);
}

template <typename T, int width>
typename std::enable_if<is_ap_int<T>::value, T>::type scalar_from_bits(const ap_uint<width>& bits) {
    return T(bits);
}

template <typename T, int width>
typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value && !std::is_same<T, bool>::value, T>::type
scalar_from_bits(const ap_uint<width>& bits) {
    return static_cast<T>(bits);
}

template <typename T, int width>
typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type scalar_from_bits(
    const ap_uint<width>& bits) {
    const ap_int<width> signed_bits = bits;
    return static_cast<T>(signed_bits);
}

template <typename T, int word_bw>
T read_scalar_value(const ap_uint<word_bw>* src, ReadCursor<word_bw>& cursor) {
    constexpr int elem_bw = bit_width<T>::value;
    const ap_uint<elem_bw> bits = read_packed_bits<elem_bw>(src, cursor);
    return scalar_from_bits<T>(bits);
}

template <typename T, int word_bw>
void read_schema_via_recursive(const ap_uint<word_bw>* src, ReadCursor<word_bw>& cursor, T& value) {
    if constexpr (has_read_recursive_inplace<T, word_bw>::value) {
        T::template read_recursive<word_bw>(src, cursor.word_idx, cursor.bit_idx, value);
    }
    else {
        value = T::template read_recursive<word_bw>(src, cursor.word_idx, cursor.bit_idx);
    }
}

template <typename T, int word_bw>
void read_schema_via_unpack(const ap_uint<word_bw>* src, ReadCursor<word_bw>& cursor, T& value) {
    constexpr int elem_bw = T::bitwidth;
    const ap_uint<elem_bw> bits = read_packed_bits<elem_bw>(src, cursor);
    value = T::unpack_from_uint(bits);
}

template <typename T, int word_bw>
void read_one(const ap_uint<word_bw>* src, ReadCursor<word_bw>& cursor, T& value) {
    if constexpr (is_supported_scalar<T>::value) {
        value = read_scalar_value<T, word_bw>(src, cursor);
    }
    else if constexpr (has_read_recursive_inplace<T, word_bw>::value || has_read_recursive_return<T, word_bw>::value) {
        read_schema_via_recursive<T, word_bw>(src, cursor, value);
    }
    else if constexpr (has_schema_unpack_api<T>::value) {
        read_schema_via_unpack<T, word_bw>(src, cursor, value);
    }
    else {
        static_assert(
            is_supported_scalar<T>::value || has_schema_unpack_api<T>::value ||
                has_read_recursive_inplace<T, word_bw>::value || has_read_recursive_return<T, word_bw>::value,
            "array_utils::read_array requires either a supported scalar type or a schema type with "
            "static read_recursive<word_bw>(...), or static constexpr bitwidth plus unpack_from_uint(...)."
        );
    }
}

}  // namespace detail

/**
 * @brief Unpack a densely packed array from an ap_uint word buffer.
 *
 * The source buffer is interpreted using the same greedy, LSB-first packing convention
 * emitted by PySilicon schema generators: elements are packed into each word from bit 0
 * upward, and if the next element does not fit in the remaining bits of the current word,
 * decoding resumes at bit 0 of the next word. Scalar element types are reconstructed
 * directly from the packed bits. Schema element types are decoded by first preferring a
 * static `read_recursive<word_bw>` API when one is available; otherwise the helper falls
 * back to `T::bitwidth` plus `T::unpack_from_uint(...)`, which matches the current
 * generated schema headers.
 *
 * @tparam T Element type to decode. Supported scalar types are `float`, built-in signed and
 * unsigned integer types, `ap_uint<N>`, and `ap_int<N>`. Schema types must provide either a
 * static `read_recursive<word_bw>` overload that advances a word/bit cursor, or a static
 * `constexpr int bitwidth` and `unpack_from_uint(...)`.
 * @tparam word_bw Width of each packed source word in bits.
 * @param src Pointer to the packed source words.
 * @param dst Pointer to the destination array.
 * @param shape Number of elements to decode.
 */
template <typename T, int word_bw>
void read_array(const ap_uint<word_bw>* src, T* dst, int shape) {
    static_assert(word_bw > 0, "word_bw must be positive.");

    if (src == nullptr || dst == nullptr || shape <= 0) {
        return;
    }

    detail::ReadCursor<word_bw> cursor;
    for (int idx = 0; idx < shape; ++idx) {
        detail::read_one<T, word_bw>(src, cursor, dst[idx]);
    }
}

}  // namespace array_utils

#endif  // PYSILICON_CODEGEN_ARRAY_UTILS_H