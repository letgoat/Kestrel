

#pargma once

#include <vector>
#include <string>

namespace kestrel {


    enum class GGUFValueType : uint32_t {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
    };

    enum class GGMLType : uint32_t {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_I16     = 25,
        GGML_TYPE_I32     = 26,
        GGML_TYPE_I64     = 27,
        GGML_TYPE_F64     = 28,
        GGML_TYPE_IQ1_M   = 29,
        GGML_TYPE_BF16    = 30,
        // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        // GGML_TYPE_Q4_0_4_8 = 32,
        // GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
        // GGML_TYPE_IQ4_NL_4_4 = 36,
        // GGML_TYPE_IQ4_NL_4_8 = 37,
        // GGML_TYPE_IQ4_NL_8_8 = 38,
        GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
        GGML_TYPE_COUNT   = 40,

        UNKNOWN = 0xFFFFFFFFu,
    }


    // 元数据metadata的值的struct
    struct GGUFValue {
        GGUFValueType type = GGUFValueType::UINT8;

        union {
            uint8_t  u8;
            int8_t   i8;
            uint16_t u16;
            int16_t  i16;
            uint32_t u32;
            int32_t  i32;
            float    f32;
            bool     b;
            uint64_t u64;
            int64_t  i64;
            double   f64;
        } scalar;

        std::string str;
        std::vector<GGUFValue> arr;

        GGUFValue() = default;

        static GGUFValue from_u8(uint8_t v) {GGUFValue tmp; tmp.type = GGUFValueType::UINT8; tmp.scalar.u8 = v; return tmp;}
        static GGUFValue from_i8(int8_t v) {GGUFValue tmp; tmp.type = GGUFValueType::INT8; tmp.scalar.i8 = v; return tmp;}
        static GGUFValue from_u16(uint16_t v) {GGUFValue tmp; tmp.type = GGUFValueType::UINT16; tmp.scalar.u16 = v; return tmp;}
        static GGUFValue from_i16(int16_t v) {GGUFValue tmp; tmp.type = GGUFValueType::INT16; tmp.scalar.i16 = v; return tmp;}
        static GGUFValue from_u32(uint32_t v) {GGUFValue tmp; tmp.type = GGUFValueType::UINT32; tmp.scalar.u32 = v; return tmp;}
        static GGUFValue from_i32(int32_t v) {GGUFValue tmp; tmp.type = GGUFValueType::INT32; tmp.scalar.i32 = v; return tmp;}
        static GGUFValue from_f32(float v) {GGUFValue tmp; tmp.type = GGUFValueType::FLOAT32; tmp.scalar.f32 = v; return tmp;}
        static GGUFValue from_bool(bool v) {GGUFValue tmp; tmp.type = GGUFValueType::BOOL; tmp.scalar.b = v; return tmp;}
        static GGUFValue from_u64(uint64_t v) {GGUFValue tmp; tmp.type = GGUFValueType::UINT64; tmp.scalar.u64 = v; return tmp;}
        static GGUFValue from_i64(int64_t v) {GGUFValue tmp; tmp.type = GGUFValueType::INT64; tmp.scalar.i64 = v; return tmp;}
        static GGUFValue from_f64(double v) {GGUFValue tmp; tmp.type = GGUFValueType::FLOAT64; tmp.scalar.f64 = v; return tmp;}
        static GGUFValue from_string(std::string s) {GGUFValue tmp; tmp.type = GGUFValueType::STRING; tmp.str = std::move(s); return tmp;}
        static GGUFValue from_array(std::vector<GGUFValue> a) {GGUFValue tmp; tmp.type = GGUFValueType::ARRAY; tmp.arr = std::move(a); return tmp;}
    }

    // 张量信息tensor info的struct
    struct TensorInfo {
        GGMLType dtype;
        std::string name;
        std::vector<uint64_t> shape;
        uint64_t offset;

        // 计算张量元素总数（所有维度之积）
        uint64_t num_elements() const {
            uint64_t n = 1;
            for (auto d : shape) n *= d;
            return n;
        }
    }

    // GGUF 文件头 Magic number（小端序存储为 0x46554747）
    inline constexpr uint32_t GGUF_MAGIC   = 0x46554747u;  // 'G','G','U','F'
}