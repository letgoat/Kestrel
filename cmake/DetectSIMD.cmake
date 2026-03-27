include(CheckCXXSourceRuns)

# ── Detect AVX2 support ─────────────────────────────────────────────────────
set(_avx2_test_src "
#include <immintrin.h>
int main() {
    __m256i a = _mm256_set1_epi32(1);
    __m256i b = _mm256_set1_epi32(2);
    __m256i c = _mm256_add_epi32(a, b);
    (void)c;
    return 0;
}
")

set(CMAKE_REQUIRED_FLAGS "-mavx2")
check_cxx_source_runs("${_avx2_test_src}" KESTREL_HAS_AVX2)
unset(CMAKE_REQUIRED_FLAGS)

if(KESTREL_HAS_AVX2)
    message(STATUS "[Kestrel] AVX2 support detected — enabling -mavx2")
    add_compile_options(-mavx2)
    add_compile_definitions(KESTREL_HAS_AVX2=1)
else()
    message(STATUS "[Kestrel] AVX2 not available on this machine")
    add_compile_definitions(KESTREL_HAS_AVX2=0)
endif()
