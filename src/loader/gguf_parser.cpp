/**
 * @file gguf_parser.cpp
 * @brief GGUFParser 类实现 —— GGUF 文件完整解析
 *
 * 实现约定：
 *   · 使用 mmap(PROT_READ, MAP_PRIVATE) 映射文件，所有读取通过 memcpy 完成，避免未对齐访问 UB。
 *   · 全部解析操作假定文件为小端序（GGUF 规范强制要求）。
 *   · 解析失败时抛出 std::runtime_error，不使用错误码返回值。
 */

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kestrel/loader/gguf_parser.h"
#include "kestrel/loader/gguf_types.h"
// #include "kestrel/types.h"   // ModelConfig（Week 1 Day 5 实现后取消注释）

namespace kestrel {

// ─────────────────────────────────────────────────────────────────────────────
// 构造 / 析构
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 打开文件、获取大小、建立只读 mmap 映射。
 * @throws std::runtime_error  open / fstat / mmap 失败时抛出。
 */
GGUFParser::GGUFParser(const std::string& path)
    : path_(path)
{   
    fd_ = open(path.c_str(), O_RDONLY) 
    if (fd_ == -1) {
        throw std::runtime_error("open file failed");
    }
    file_size_ = lseek(fd_, 0, SEEK_END);
    if (file_size_ == -1) {
        throw std::runtime_error("lseek failed");
    }
    data_ = (const char*)mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        throw std::runtime_error("mmap failed");
    }
}

/**
 * 释放 mmap 映射并关闭文件描述符。
 */
GGUFParser::~GGUFParser()
{
    // TODO: 实现
    // 1. if (data_ && data_ != MAP_FAILED) munmap(const_cast<char*>(data_), file_size_)
    // 2. if (fd_ != -1) close(fd_)
}

/**
 * 移动构造：转移 OS 资源所有权，将源对象置为"空"状态。
 */
GGUFParser::GGUFParser(GGUFParser&& other) noexcept
{
    // TODO: 实现
    // 将 other 的所有成员转移到 *this，然后将 other 重置为安全的默认值
    // 例如：path_ = std::move(other.path_); fd_ = other.fd_; other.fd_ = -1; ...
}

/**
 * 移动赋值：先释放自身资源，再转移 other 的资源。
 */
GGUFParser& GGUFParser::operator=(GGUFParser&& other) noexcept
{
    // TODO: 实现
    // 1. 防自赋值检查：if (this == &other) return *this;
    // 2. 释放已有资源（同析构逻辑）
    // 3. 转移 other 的所有成员，并将 other 重置
    return *this;
}

// ─────────────────────────────────────────────────────────────────────────────
// 主解析入口
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 依次执行 parse_header()、parse_metadata()、parse_tensor_infos()，
 * 并在结束后计算数据区偏移（含对齐）。
 * 重复调用时先清空上次结果。
 */
void GGUFParser::parse()
{
    // TODO: 实现
    // 1. 重置状态：pos_ = 0; metadata_.clear(); tensors_.clear();
    // 2. parse_header();
    // 3. parse_metadata();
    // 4. parse_tensor_infos();
    // 5. 读取 alignment（metadata_ 中的 "general.alignment"，缺省 GGUF_DEFAULT_ALIGNMENT）
    // 6. align_offset(alignment)，将对齐后的 pos_ 赋给 data_section_offset_
}

// ─────────────────────────────────────────────────────────────────────────────
// Header 字段访问
// ─────────────────────────────────────────────────────────────────────────────

uint32_t GGUFParser::version() const noexcept
{
    return version_;
}

uint64_t GGUFParser::n_tensors() const noexcept
{
    return n_tensors_;
}

uint64_t GGUFParser::n_kv() const noexcept
{
    return n_kv_;
}

// ─────────────────────────────────────────────────────────────────────────────
// 元数据访问
// ─────────────────────────────────────────────────────────────────────────────

const std::unordered_map<std::string, GGUFValue>& GGUFParser::metadata() const noexcept
{
    return metadata_;
}

bool GGUFParser::has_key(const std::string& key) const noexcept
{
    return metadata_.count(key) != 0;
}

/**
 * @throws std::out_of_range   key 不存在。
 * @throws std::runtime_error  类型不匹配。
 */
const std::string& GGUFParser::get_string(const std::string& key) const
{
    // TODO: 实现
    // 1. 查找 key，不存在则 throw std::out_of_range
    // 2. 检查 value.type == GGUFValueType::STRING，否则 throw std::runtime_error
    // 3. return value.str
    throw std::runtime_error("not implemented");
}

/**
 * @throws std::out_of_range   key 不存在。
 * @throws std::runtime_error  类型不匹配。
 */
uint32_t GGUFParser::get_uint32(const std::string& key) const
{
    // TODO: 实现
    // 类似 get_string，检查 GGUFValueType::UINT32，返回 value.scalar.u32
    throw std::runtime_error("not implemented");
}

/**
 * @throws std::out_of_range   key 不存在。
 * @throws std::runtime_error  类型不匹配。
 */
uint64_t GGUFParser::get_uint64(const std::string& key) const
{
    // TODO: 实现
    // 检查 GGUFValueType::UINT64，返回 value.scalar.u64
    throw std::runtime_error("not implemented");
}

/**
 * @throws std::out_of_range   key 不存在。
 * @throws std::runtime_error  类型不匹配。
 */
int32_t GGUFParser::get_int32(const std::string& key) const
{
    // TODO: 实现
    // 检查 GGUFValueType::INT32，返回 value.scalar.i32
    throw std::runtime_error("not implemented");
}

/**
 * @throws std::out_of_range   key 不存在。
 * @throws std::runtime_error  类型不匹配。
 */
float GGUFParser::get_float32(const std::string& key) const
{
    // TODO: 实现
    // 检查 GGUFValueType::FLOAT32，返回 value.scalar.f32
    throw std::runtime_error("not implemented");
}

/**
 * @throws std::out_of_range   key 不存在。
 * @throws std::runtime_error  类型不匹配。
 */
bool GGUFParser::get_bool(const std::string& key) const
{
    // TODO: 实现
    // 检查 GGUFValueType::BOOL，返回 value.scalar.b
    throw std::runtime_error("not implemented");
}

/**
 * @throws std::out_of_range   key 不存在。
 * @throws std::runtime_error  类型不匹配。
 */
const std::vector<GGUFValue>& GGUFParser::get_array(const std::string& key) const
{
    // TODO: 实现
    // 检查 GGUFValueType::ARRAY，返回 value.arr
    throw std::runtime_error("not implemented");
}

// ─────────────────────────────────────────────────────────────────────────────
// 张量描述符访问
// ─────────────────────────────────────────────────────────────────────────────

const std::vector<TensorInfo>& GGUFParser::tensors() const noexcept
{
    return tensors_;
}

const TensorInfo* GGUFParser::find_tensor(const std::string& name) const noexcept
{
    // TODO: 实现
    // 线性遍历 tensors_，找到 name 相同的返回其地址，否则返回 nullptr
    return nullptr;
}

uint64_t GGUFParser::data_section_offset() const noexcept
{
    return data_section_offset_;
}

uint64_t GGUFParser::data_section_size() const noexcept
{
    // TODO: 实现
    // return file_size_ - data_section_offset_
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// 结构化配置提取
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 根据 "general.architecture" 自动选择前缀（"qwen2" / "llama"），
 * 读取对应字段并填充 ModelConfig 结构体。
 * @throws std::runtime_error  缺少必需字段或 architecture 不支持时抛出。
 */
ModelConfig GGUFParser::extract_config() const
{
    // TODO: 实现（依赖 kestrel/types.h 中的 ModelConfig，Week 1 Day 5 后完成）
    // 1. arch = get_string("general.architecture")
    // 2. prefix = arch + "."
    // 3. 读取 prefix + "context_length"、"embedding_length"、"feed_forward_length"、
    //    "attention.head_count"、"attention.head_count_kv"、"block_count" 等字段
    // 4. 填充并返回 ModelConfig
    throw std::runtime_error("extract_config: not implemented");
}

// ─────────────────────────────────────────────────────────────────────────────
// 分步解析（private）
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 解析 24 字节文件头：magic（4B）、version（4B）、n_tensors（8B）、n_kv（8B）。
 * @throws std::runtime_error  magic 不匹配、版本不支持或文件过短时抛出。
 */
void GGUFParser::parse_header()
{
    // TODO: 实现
    // 1. uint32_t magic = read_u32();
    //    if (magic != GGUF_MAGIC) throw std::runtime_error("invalid GGUF magic");
    // 2. version_ = read_u32();
    //    if (version_ < 1 || version_ > 3) throw std::runtime_error("unsupported GGUF version");
    // 3. n_tensors_ = read_u64();
    // 4. n_kv_      = read_u64();
}

/**
 * 循环 n_kv_ 次，每次读取一个 KV 对（key 字符串 + value_type + value），
 * 存入 metadata_。
 */
void GGUFParser::parse_metadata()
{
    // TODO: 实现
    // for (uint64_t i = 0; i < n_kv_; ++i) {
    //     std::string key = read_gguf_string();
    //     GGUFValueType vtype = static_cast<GGUFValueType>(read_u32());
    //     GGUFValue val = read_value(vtype);
    //     metadata_.emplace(std::move(key), std::move(val));
    // }
}

/**
 * 循环 n_tensors_ 次，每次读取一个 TensorInfo，存入 tensors_。
 * TensorInfo 格式：name（gguf_string）、n_dims（u32）、shape（n_dims × u64）、
 *                  dtype（u32）、offset（u64）。
 */
void GGUFParser::parse_tensor_infos()
{
    // TODO: 实现
    // tensors_.reserve(n_tensors_);
    // for (uint64_t i = 0; i < n_tensors_; ++i) {
    //     TensorInfo ti;
    //     ti.name   = read_gguf_string();
    //     uint32_t n_dims = read_u32();
    //     ti.shape.resize(n_dims);
    //     for (auto& d : ti.shape) d = read_u64();
    //     ti.dtype  = static_cast<GGMLType>(read_u32());
    //     ti.offset = read_u64();
    //     tensors_.push_back(std::move(ti));
    // }
}

// ─────────────────────────────────────────────────────────────────────────────
// 底层读取工具（private）
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 从 data_[pos_] 读取 n 字节到 dst，然后 pos_ += n。
 * @throws std::runtime_error  pos_ + n > file_size_ 时抛出。
 */
void GGUFParser::read_bytes(void* dst, size_t n)
{
    // TODO: 实现
    // if (pos_ + n > file_size_) throw std::runtime_error("unexpected end of file");
    // std::memcpy(dst, data_ + pos_, n);
    // pos_ += n;
}

uint8_t GGUFParser::read_u8()
{
    uint8_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

uint16_t GGUFParser::read_u16()
{
    uint16_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

uint32_t GGUFParser::read_u32()
{
    uint32_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

uint64_t GGUFParser::read_u64()
{
    uint64_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

int8_t GGUFParser::read_i8()
{
    int8_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

int16_t GGUFParser::read_i16()
{
    int16_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

int32_t GGUFParser::read_i32()
{
    int32_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

int64_t GGUFParser::read_i64()
{
    int64_t v;
    read_bytes(&v, sizeof(v));
    return v;
}

float GGUFParser::read_f32()
{
    float v;
    read_bytes(&v, sizeof(v));
    return v;
}

double GGUFParser::read_f64()
{
    double v;
    read_bytes(&v, sizeof(v));
    return v;
}

/**
 * 读取 GGUF 字符串：先读 uint64 长度，再读对应字节数的 UTF-8 内容。
 * @throws std::runtime_error  长度字段超出剩余文件范围时抛出。
 */
std::string GGUFParser::read_gguf_string()
{
    // TODO: 实现
    // uint64_t len = read_u64();
    // if (pos_ + len > file_size_) throw std::runtime_error("string length out of bounds");
    // std::string s(data_ + pos_, len);
    // pos_ += len;
    // return s;
    return {};
}

/**
 * 根据 vtype 分派读取逻辑；ARRAY 类型先读 elem_type 和 count，再递归读取每个元素。
 * @throws std::runtime_error  遇到未知 vtype 时抛出。
 */
GGUFValue GGUFParser::read_value(GGUFValueType vtype)
{
    // TODO: 实现
    // switch (vtype) {
    //   case GGUFValueType::UINT8:   return GGUFValue::from_u8(read_u8());
    //   case GGUFValueType::INT8:    return GGUFValue::from_i8(read_i8());
    //   case GGUFValueType::UINT16:  return GGUFValue::from_u16(read_u16());
    //   case GGUFValueType::INT16:   return GGUFValue::from_i16(read_i16());
    //   case GGUFValueType::UINT32:  return GGUFValue::from_u32(read_u32());
    //   case GGUFValueType::INT32:   return GGUFValue::from_i32(read_i32());
    //   case GGUFValueType::FLOAT32: return GGUFValue::from_f32(read_f32());
    //   case GGUFValueType::BOOL:    return GGUFValue::from_bool(read_u8() != 0);
    //   case GGUFValueType::STRING:  return GGUFValue::from_string(read_gguf_string());
    //   case GGUFValueType::UINT64:  return GGUFValue::from_u64(read_u64());
    //   case GGUFValueType::INT64:   return GGUFValue::from_i64(read_i64());
    //   case GGUFValueType::FLOAT64: return GGUFValue::from_f64(read_f64());
    //   case GGUFValueType::ARRAY: {
    //       GGUFValueType elem_type = static_cast<GGUFValueType>(read_u32());
    //       uint64_t count = read_u64();
    //       std::vector<GGUFValue> arr;
    //       arr.reserve(count);
    //       for (uint64_t i = 0; i < count; ++i)
    //           arr.push_back(read_value(elem_type));
    //       return GGUFValue::from_array(std::move(arr));
    //   }
    //   default:
    //       throw std::runtime_error("unknown GGUFValueType: " + std::to_string((uint32_t)vtype));
    // }
    return {};
}

/**
 * 将 pos_ 向前对齐到 alignment 字节边界（pos_ = ceil(pos_ / alignment) * alignment）。
 */
void GGUFParser::align_offset(uint32_t alignment) noexcept
{
    // TODO: 实现
    // if (alignment == 0) return;
    // pos_ = (pos_ + alignment - 1) / alignment * alignment;
}

} // namespace kestrel
