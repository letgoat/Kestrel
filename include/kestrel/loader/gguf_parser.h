/**
 * @file gguf_parser.h
 * @brief GGUFParser 类声明 —— GGUF 文件的完整解析器接口
 *
 * 用法示例：
 * @code
 *   kestrel::GGUFParser parser("/path/to/model.gguf");
 *   parser.parse();
 *
 *   std::cout << "architecture: " << parser.get_string("general.architecture") << "\n";
 *   for (const auto& t : parser.tensors()) {
 *       std::cout << t.name << " dtype=" << (uint32_t)t.dtype << "\n";
 *   }
 * @endcode
 *
 * 实现说明（gguf_parser.cpp）：
 *   · 使用 mmap(PROT_READ, MAP_PRIVATE) 映射文件，所有读取通过 memcpy 完成，
 *     避免未对齐访问的 UB。
 *   · 全部解析操作假定文件为小端序（GGUF 规范强制要求）。
 *   · 解析失败时抛出 std::runtime_error，不使用错误码返回值。
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "kestrel/loader/gguf_types.h"

// ModelConfig 在 include/kestrel/types.h 中定义（Week 1 Day 5 实现）
namespace kestrel { struct ModelConfig; }

namespace kestrel {

/**
 * @class GGUFParser
 * @brief 将 GGUF 文件解析为结构化的元数据和张量描述符。
 *
 * 生命周期：
 *   1. 构造时打开文件并建立 mmap 映射；
 *   2. 调用 parse() 完成全部解析（或分步调用各 parse_* 方法）；
 *   3. 析构时自动 munmap 并关闭文件描述符。
 *
 * 线程安全性：不可在多线程间共享同一个 GGUFParser 实例（无内部锁）。
 */
class GGUFParser {
public:
    // ── 构造 / 析构 ──────────────────────────────────────────────────────────

    /**
     * @brief 打开 GGUF 文件并建立只读 mmap 映射。
     * @param path  GGUF 文件的绝对或相对路径。
     * @throws std::runtime_error  文件不存在或无读取权限时抛出。
     */
    explicit GGUFParser(const std::string& path);

    /**
     * @brief 析构函数，释放 mmap 映射并关闭文件描述符。
     */
    ~GGUFParser();

    // 禁止拷贝（持有 OS 资源）
    GGUFParser(const GGUFParser&)            = delete;
    GGUFParser& operator=(const GGUFParser&) = delete;

    // 允许移动
    GGUFParser(GGUFParser&&)            noexcept;
    GGUFParser& operator=(GGUFParser&&) noexcept;

    // ── 主解析入口 ───────────────────────────────────────────────────────────

    /**
     * @brief 完整解析文件：依次执行 parse_header()、parse_metadata()、parse_tensor_infos()。
     *
     * 调用后可通过下方的访问接口读取所有解析结果。
     * 同一个对象重复调用会先清空上次的结果再重新解析。
     *
     * @throws std::runtime_error  Magic 不匹配、版本不支持、数据截断等格式错误时抛出。
     */
    void parse();

    // ── Header 字段访问 ──────────────────────────────────────────────────────

    /**
     * @brief 返回 GGUF 文件版本（目前已知 v1、v2、v3）。
     * @pre   parse() 或 parse_header() 已调用。
     */
    uint32_t version() const noexcept;

    /**
     * @brief 返回文件中的张量总数（n_tensors 字段）。
     * @pre   parse() 或 parse_header() 已调用。
     */
    uint64_t n_tensors() const noexcept;

    /**
     * @brief 返回元数据 KV 对总数（n_kv 字段）。
     * @pre   parse() 或 parse_header() 已调用。
     */
    uint64_t n_kv() const noexcept;

    // ── 元数据访问 ───────────────────────────────────────────────────────────

    /**
     * @brief 返回全部元数据，key 为 GGUF 规范中的 key 字符串（如 "general.architecture"）。
     * @pre   parse() 或 parse_metadata() 已调用。
     * @return 只读引用，生命周期与本 GGUFParser 对象相同。
     */
    const std::unordered_map<std::string, GGUFValue>& metadata() const noexcept;

    /**
     * @brief 检查元数据中是否存在指定 key。
     * @pre   parse() 或 parse_metadata() 已调用。
     */
    bool has_key(const std::string& key) const noexcept;

    /**
     * @brief 读取 STRING 类型的元数据值。
     * @throws std::out_of_range   key 不存在时抛出。
     * @throws std::runtime_error  key 存在但类型不是 STRING 时抛出。
     */
    const std::string& get_string(const std::string& key) const;

    /**
     * @brief 读取 UINT32 类型的元数据值。
     * @throws std::out_of_range   key 不存在时抛出。
     * @throws std::runtime_error  key 存在但类型不是 UINT32 时抛出。
     */
    uint32_t get_uint32(const std::string& key) const;

    /**
     * @brief 读取 UINT64 类型的元数据值。
     * @throws std::out_of_range   key 不存在时抛出。
     * @throws std::runtime_error  key 存在但类型不是 UINT64 时抛出。
     */
    uint64_t get_uint64(const std::string& key) const;

    /**
     * @brief 读取 INT32 类型的元数据值。
     * @throws std::out_of_range   key 不存在时抛出。
     * @throws std::runtime_error  key 存在但类型不是 INT32 时抛出。
     */
    int32_t get_int32(const std::string& key) const;

    /**
     * @brief 读取 FLOAT32 类型的元数据值。
     * @throws std::out_of_range   key 不存在时抛出。
     * @throws std::runtime_error  key 存在但类型不是 FLOAT32 时抛出。
     */
    float get_float32(const std::string& key) const;

    /**
     * @brief 读取 BOOL 类型的元数据值。
     * @throws std::out_of_range   key 不存在时抛出。
     * @throws std::runtime_error  key 存在但类型不是 BOOL 时抛出。
     */
    bool get_bool(const std::string& key) const;

    /**
     * @brief 读取 ARRAY 类型的元数据值。
     * @throws std::out_of_range   key 不存在时抛出。
     * @throws std::runtime_error  key 存在但类型不是 ARRAY 时抛出。
     */
    const std::vector<GGUFValue>& get_array(const std::string& key) const;

    // ── 张量描述符访问 ───────────────────────────────────────────────────────

    /**
     * @brief 返回全部张量描述符，顺序与文件中一致。
     * @pre   parse() 或 parse_tensor_infos() 已调用。
     * @return 只读引用，生命周期与本 GGUFParser 对象相同。
     */
    const std::vector<TensorInfo>& tensors() const noexcept;

    /**
     * @brief 按名称查找张量描述符。
     * @return 指向目标 TensorInfo 的指针；若不存在则返回 nullptr。
     * @pre   parse() 或 parse_tensor_infos() 已调用。
     */
    const TensorInfo* find_tensor(const std::string& name) const noexcept;

    /**
     * @brief 返回张量数据区在文件中的绝对字节偏移。
     *
     * 数据区起始 = tensor info 段结束后对齐到 alignment 的第一个字节。
     * alignment 优先使用元数据键 "general.alignment"，缺省为 GGUF_DEFAULT_ALIGNMENT（32）。
     *
     * @pre   parse() 已调用。
     */
    uint64_t data_section_offset() const noexcept;

    /**
     * @brief 返回数据区总字节数（= 文件大小 - data_section_offset()）。
     * @pre   parse() 已调用。
     */
    uint64_t data_section_size() const noexcept;

    // ── 结构化配置提取 ───────────────────────────────────────────────────────

    /**
     * @brief 从元数据中提取结构化的模型配置（ModelConfig）。
     *
     * 根据 "general.architecture" 键自动选择字段前缀：
     *   - "qwen2"  → 读取 qwen2.* 字段
     *   - "llama"  → 读取 llama.* 字段
     *
     * @pre   parse() 已调用；kestrel/types.h 中的 ModelConfig 已定义（Day 5 实现）。
     * @throws std::runtime_error  缺少必需字段或 architecture 不支持时抛出。
     */
    ModelConfig extract_config() const;

private:
    // ── 分步解析（parse() 内部依次调用） ────────────────────────────────────

    /// 解析 24 字节的文件头（magic、version、n_tensors、n_kv）
    void parse_header();

    /// 解析所有元数据 KV 对，结果写入 metadata_
    void parse_metadata();

    /// 解析所有张量描述符，结果写入 tensors_
    void parse_tensor_infos();

    // ── 底层读取工具 ─────────────────────────────────────────────────────────

    /**
     * @brief 从当前偏移读取 n 字节到 dst，并将偏移前进 n。
     * @throws std::runtime_error  超出文件边界时抛出。
     */
    void read_bytes(void* dst, size_t n);

    /// 读取各基础类型（均使用 memcpy，避免未对齐 UB）
    uint8_t  read_u8();
    uint16_t read_u16();
    uint32_t read_u32();
    uint64_t read_u64();
    int8_t   read_i8();
    int16_t  read_i16();
    int32_t  read_i32();
    int64_t  read_i64();
    float    read_f32();
    double   read_f64();

    /**
     * @brief 读取 GGUF 字符串（uint64 长度前缀 + UTF-8 字节序列）。
     * @throws std::runtime_error  长度字段超出剩余文件大小时抛出。
     */
    std::string read_gguf_string();

    /**
     * @brief 根据 GGUFValueType 读取对应的值，ARRAY 类型会递归调用自身。
     * @throws std::runtime_error  遇到未知类型时抛出。
     */
    GGUFValue read_value(GGUFValueType vtype);

    /// 将当前偏移向前对齐到 alignment 字节边界（跳过填充字节）
    void align_offset(uint32_t alignment) noexcept;

    // ── 内部状态 ─────────────────────────────────────────────────────────────

    std::string path_;            ///< 文件路径（保留用于错误信息）
    int         fd_   = -1;       ///< open() 返回的文件描述符
    const char* data_ = nullptr;  ///< mmap 返回的只读映射基址
    size_t      file_size_ = 0;   ///< 文件字节数
    size_t      pos_       = 0;   ///< 当前读取游标（相对于 data_ 的偏移）

    // Header 字段
    uint32_t version_   = 0;
    uint64_t n_tensors_ = 0;
    uint64_t n_kv_      = 0;

    // 解析结果
    std::unordered_map<std::string, GGUFValue> metadata_;
    std::vector<TensorInfo>                    tensors_;
    uint64_t data_section_offset_ = 0;  ///< 张量数据区绝对偏移（对齐后）
};

} // namespace kestrel
