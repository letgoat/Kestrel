add_executable(gguf_dump main.cpp)

target_compile_features(gguf_dump PRIVATE cxx_std_17)

set_target_properties(gguf_dump PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools/gguf_dump
)
