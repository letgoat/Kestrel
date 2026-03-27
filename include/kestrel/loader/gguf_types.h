/**
 * @file gguf_types.h
 * @brief GGUF 格式核心枚举与数据类型定义（纯头文件，不依赖任何 .cpp）
 *
 * 参考规范：https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 * 参考实现：llama.cpp gguf-py/gguf/constants.py
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kestrel {

// ─────────────────────────────────────────────────────────────────────────────
// GGUFValueType — 元数据 KV 对的值类型（value_type 字段，uint32 存储）
// 枚举值与 GGUF 规范中的数值一一对应，不可随意修改顺序。
// ─────────────────────────────────────────────────────────────────────────────

enum class GGUFValueType : uint32_t {
    UINT8   = 0,   ///< uint8_t
    INT8    = 1,   ///< int8_t
    UINT16  = 2,   ///< uint16_t
    INT16   = 3,   ///< int16_t
    UINT32  = 4,   ///< uint32_t
    INT32   = 5,   ///< int32_t
    FLOAT32 = 6,   ///< float
    BOOL    = 7,   ///< uint8_t，0 = false，非零 = true
    STRING  = 8,   ///< gguf_string：uint64(len) + bytes(len)，UTF-8
    ARRAY   = 9,   ///< uint32(elem_type) + uint64(count) + elem×count
    UINT64  = 10,  ///< uint64_t
    INT64   = 11,  ///< int64_t
    FLOAT64 = 12,  ///< double
};

// ─────────────────────────────────────────────────────────────────────────────
// GGMLType — 张量数据类型（tensor info 中的 dtype 字段，uint32 存储）
// 数值与 ggml/include/ggml.h 中的 ggml_type 枚举完全一致，确保与 llama.cpp 互操作。
// ─────────────────────────────────────────────────────────────────────────────

enum class GGMLType : uint32_t {
    F32    = 0,   ///< 32-bit 浮点（全精度基准）
    F16    = 1,   ///< 16-bit 半精度浮点（推理/存储常用）
    Q4_0   = 2,   ///< 4-bit 量化，无 scale 偏移，block=32
    Q4_1   = 3,   ///< 4-bit 量化，有 min 偏移，block=32
    Q5_0   = 6,   ///< 5-bit 量化，无 scale 偏移，block=32
    Q5_1   = 7,   ///< 5-bit 量化，有 min 偏移，block=32
    Q8_0   = 8,   ///< 8-bit 量化，带 block scale，block=32（反量化速度快）
    Q4_K   = 12,  ///< 4-bit K-quant，block=256，精度/压缩率均衡（Kestrel 主目标）
    Q6_K   = 14,  ///< 6-bit K-quant，block=256，接近 F16 精度
    Q8_K   = 15,  ///< 8-bit K-quant，block=256，仅用于中间量化结果
    BF16   = 30,  ///< Brain Float 16，指数位与 F32 相同，训练/推理均支持

    // 未知类型占位符，解析到未识别的值时使用
    UNKNOWN = 0xFFFFFFFFu,
};

// ─────────────────────────────────────────────────────────────────────────────
// GGUFValue — 元数据值的容器（标量 + string + array 的统一表示）
//
// 设计说明：
//   · 采用"类型标签 + 联合体 + 独立 string/array 成员"的方式，
//     避免 C++17 recursive variant 的 incomplete-type 限制。
//   · 只读场景下不涉及性能热路径，清晰性优先于极致性能。
// ─────────────────────────────────────────────────────────────────────────────

struct GGUFValue {
    GGUFValueType type = GGUFValueType::UINT8;

    /// 标量存储（UINT8 ~ FLOAT64，除 STRING/ARRAY/BOOL 外的 10 种）
    union Scalar {
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

        Scalar() : u64(0) {}
    } scalar;

    std::string          str;   ///< 仅 STRING 类型使用
    std::vector<GGUFValue> arr; ///< 仅 ARRAY 类型使用，元素类型均相同

    // ── 便捷构造函数 ──────────────────────────────────────────────────────────
    GGUFValue() = default;

    static GGUFValue from_u8(uint8_t v)   { GGUFValue r; r.type = GGUFValueType::UINT8;   r.scalar.u8  = v; return r; }
    static GGUFValue from_i8(int8_t v)    { GGUFValue r; r.type = GGUFValueType::INT8;    r.scalar.i8  = v; return r; }
    static GGUFValue from_u16(uint16_t v) { GGUFValue r; r.type = GGUFValueType::UINT16;  r.scalar.u16 = v; return r; }
    static GGUFValue from_i16(int16_t v)  { GGUFValue r; r.type = GGUFValueType::INT16;   r.scalar.i16 = v; return r; }
    static GGUFValue from_u32(uint32_t v) { GGUFValue r; r.type = GGUFValueType::UINT32;  r.scalar.u32 = v; return r; }
    static GGUFValue from_i32(int32_t v)  { GGUFValue r; r.type = GGUFValueType::INT32;   r.scalar.i32 = v; return r; }
    static GGUFValue from_f32(float v)    { GGUFValue r; r.type = GGUFValueType::FLOAT32; r.scalar.f32 = v; return r; }
    static GGUFValue from_bool(bool v)    { GGUFValue r; r.type = GGUFValueType::BOOL;    r.scalar.b   = v; return r; }
    static GGUFValue from_u64(uint64_t v) { GGUFValue r; r.type = GGUFValueType::UINT64;  r.scalar.u64 = v; return r; }
    static GGUFValue from_i64(int64_t v)  { GGUFValue r; r.type = GGUFValueType::INT64;   r.scalar.i64 = v; return r; }
    static GGUFValue from_f64(double v)   { GGUFValue r; r.type = GGUFValueType::FLOAT64; r.scalar.f64 = v; return r; }
    static GGUFValue from_string(std::string s)      { GGUFValue r; r.type = GGUFValueType::STRING; r.str = std::move(s); return r; }
    static GGUFValue from_array(std::vector<GGUFValue> a) { GGUFValue r; r.type = GGUFValueType::ARRAY; r.arr = std::move(a); return r; }
};

// ─────────────────────────────────────────────────────────────────────────────
// TensorInfo — 张量描述符（tensor info 段中每条记录的结构化表示）
//
// 注意：offset 是相对于数据区起始位置的字节偏移，不是文件绝对偏移。
//       数据区起始 = tensor info 段结束后，向上对齐到 GGUF_ALIGNMENT（默认 32）字节处。
// ─────────────────────────────────────────────────────────────────────────────

struct TensorInfo {
    std::string           name;    ///< 张量名称，例如 "blk.0.attn_q.weight"
    std::vector<uint64_t> shape;   ///< 各维度大小，最多 4 维；shape[0] 是最内层维度（列优先，与 numpy 相反）
    GGMLType              dtype;   ///< 数据类型（量化格式或浮点）
    uint64_t              offset;  ///< 相对于数据区起始的字节偏移

    /// 计算张量元素总数（所有维度之积）
    uint64_t num_elements() const {
        uint64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 常量
// ─────────────────────────────────────────────────────────────────────────────

/// GGUF 文件头 Magic number（小端序存储为 0x46554747）
inline constexpr uint32_t GGUF_MAGIC   = 0x46554747u;  // 'G','G','U','F'

/// 数据区默认对齐字节数；可被 general.alignment 元数据键覆盖
inline constexpr uint32_t GGUF_DEFAULT_ALIGNMENT = 32u;

} // namespace kestrel
