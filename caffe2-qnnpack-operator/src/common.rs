crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/common.h]

lazy_static!{
    /*
    #if defined(__GNUC__)
    #if defined(__clang__) || (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 5)
    #define PYTORCH_QNNP_UNREACHABLE \
      do {                           \
        __builtin_unreachable();     \
      } while (0)
    #else
    #define PYTORCH_QNNP_UNREACHABLE \
      do {                           \
        __builtin_trap();            \
      } while (0)
    #endif
    #elif defined(_MSC_VER)
    #define PYTORCH_QNNP_UNREACHABLE __assume(0)
    #else
    #define PYTORCH_QNNP_UNREACHABLE \
      do {                           \
      } while (0)
    #endif

    #if defined(_MSC_VER)
    #define PYTORCH_QNNP_ALIGN(alignment) __declspec(align(alignment))
    #else
    #define PYTORCH_QNNP_ALIGN(alignment) __attribute__((__aligned__(alignment)))
    #endif

    #define PYTORCH_QNNP_COUNT_OF(array) (sizeof(array) / sizeof(0 [array]))

    #if defined(__GNUC__)
    #define PYTORCH_QNNP_LIKELY(condition) (__builtin_expect(!!(condition), 1))
    #define PYTORCH_QNNP_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
    #else
    #define PYTORCH_QNNP_LIKELY(condition) (!!(condition))
    #define PYTORCH_QNNP_UNLIKELY(condition) (!!(condition))
    #endif

    #if defined(__GNUC__)
    #define PYTORCH_QNNP_INLINE inline __attribute__((__always_inline__))
    #else
    #define PYTORCH_QNNP_INLINE inline
    #endif

    #ifndef PYTORCH_QNNP_INTERNAL
    #if defined(__ELF__)
    #define PYTORCH_QNNP_INTERNAL __attribute__((__visibility__("internal")))
    #elif defined(__MACH__)
    #define PYTORCH_QNNP_INTERNAL __attribute__((__visibility__("hidden")))
    #else
    #define PYTORCH_QNNP_INTERNAL
    #endif
    #endif

    #ifndef PYTORCH_QNNP_PRIVATE
    #if defined(__ELF__)
    #define PYTORCH_QNNP_PRIVATE __attribute__((__visibility__("hidden")))
    #elif defined(__MACH__)
    #define PYTORCH_QNNP_PRIVATE __attribute__((__visibility__("hidden")))
    #else
    #define PYTORCH_QNNP_PRIVATE
    #endif
    #endif

    #if defined(_MSC_VER)
    #define RESTRICT_STATIC
    #define restrict
    #else
    #define RESTRICT_STATIC restrict static
    #endif

    #if defined(_MSC_VER)
    #define __builtin_prefetch
    #endif
    */
}

