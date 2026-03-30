#include <string>
#include <vector>

#include "gguf_parser_my.h"

namespace kestrel {
    GGUFParser::GGUFParser(const std::string$ path){
        path_ = path;
        fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("failed to open file: " + path);
        }
        file_size_ = lseek(fd, 0, SEEK_END);
        if (file_size == 0) {
            throw std::runtime_error("failed to get file size: " + path);
        }
    }

    GGUFParser::~GGUFParser()
    GGUFParser::GGUFParser(const GGUFParser&) = delete
    GGUFParser::GGUFParser& operator=(const GGUFParser&) = delete
    GGUFParser::GGUFParser(const GGUFParser&&)
    GGUFParser::GGUFParser& operator=(const GGUFParser&&)
    void GGUFParser::parse();
    
    uint32_t GGUFParser::version() const
    uint64_t GGUFParser::n_tensors() const
    uint64_t GGUFParser::n_kv() const
    const std::unordered_map<std::string, GGUFValue>& GGUFParser::metadata() const noexcept
    const std::vector<TensorInfo>& GGUFParser::tensors() const noexcept
    bool GGUFParser::has_key(const std::string& key) const noexcept
    const std::string& GGUFParser::get_string(const std::string& key) const
    uint32_t GGUFParser::get_uint32(const std::string& key) const
    uint64_t GGUFParser::get_uint64(const std::string& key) const
    int32_t GGUFParser::get_int32(const std::string& key) const
    int64_t GGUFParser::get_int64(const std::string& key) const
    float GGUFParser::get_float32(const std::string& key) const
    double GGUFParser::get_float64(const std::string& key) const;
    
    bool GGUFParser::get_bool(const std::string& key) const
    const std::vector<GGUFValue>& GGUFParser::get_array(const std::string& key) const
    const TensorInfo* GGUFParser::find_tensor(const std::string& name) const noexcept
    uint64_t GGUFParser::data_section_size() const noexcept;
    
    const TensorInfo* GGUFParser::find_tensor(const std::string& name) const noexcept;
}