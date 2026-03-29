
#pargma once

#include <string>
#include <vector>

namespace kestrel {

    class GGUFParser {
        public:
            GGUFParser(const std::string$ path);

            ~GGUFParser();

            GGUFParser(const GGUFParser&) = delete;

            GGUFParser& operator=(const GGUFParser&) = delete;

            GGUFParser(const GGUFParser&&);

            GGUFParser& operator=(const GGUFParser&&);

            void parse();
            
            uint32_t version() const;

            uint64_t n_tensors() const;

            uint64_t n_kv() const;

            const std::unordered_map<std::string, GGUFValue>& metadata() const noexcept;

            const std::vector<TensorInfo>& tensors() const noexcept;

            bool has_key(const std::string& key) const noexcept;

            const std::string& get_string(const std::string& key) const;

            uint32_t get_uint32(const std::string& key) const;

            uint64_t get_uint64(const std::string& key) const;

            int32_t get_int32(const std::string& key) const;

            int64_t get_int64(const std::string& key) const;

            float get_float32(const std::string& key) const;

            double get_float64(const std::string& key) const;
            
            bool get_bool(const std::string& key) const;

            const std::vector<GGUFValue>& get_array(const std::string& key) const;

            const TensorInfo* find_tensor(const std::string& name) const noexcept;

            uint64_t data_section_size() const noexcept;
            
            const TensorInfo* find_tensor(const std::string& name) const noexcept;

            
        private:
            void parse_header();
            void parse_metadata();
            void parse_tensor_infor();

            void read_bytes(void* dst, size_t n);

            std::string read_gguf_string();

            GGUFValue read_value(GGUFValueType vtype);



            uint32_t version_   = 0;
            uint64_t n_tensors_ = 0;
            uint64_t n_kv_      = 0;


            // 内部状态
            std::string path_;
            int         fd_   = -1;
            int file_size_ = 0;
            int pos_ =0;

            // 解析结果
            std::unordered_map<std::string, GGUFValue> metadata_;
            std::vector<TensorInfo> tensors_;
    }
}