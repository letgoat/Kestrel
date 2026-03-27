cmake_minimum_required(VERSION 3.21)

project(Kestrel
    VERSION 0.1.0
    DESCRIPTION "A from-scratch LLM inference engine"
    LANGUAGES CXX
)

# ── C++17 ──────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Output directories ──────────────────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ── Global include path ─────────────────────────────────────────────────────
include_directories(${PROJECT_SOURCE_DIR}/include)

# ── Optional modules ────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(DetectSIMD)
include(FetchDeps)

# ── Compiler warnings ───────────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Sub-directories ─────────────────────────────────────────────────────────
add_subdirectory(tools/gguf_dump)
