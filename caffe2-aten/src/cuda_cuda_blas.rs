// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/CUDABlas.h]

lazy_static!{
    /*
    #pragma once
    /*
      Provides a subset of CUDA BLAS functions as templates:

        gemm<Dtype>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
      ldc)

        gemv<Dtype>(transa, m, n, alpha, a, lda, x, incx, beta, y, incy)

        dot<Dtype>(n, x, incx, y, incy, result)

      where Dtype is double, float, Half or BFloat16 (ROCm, NOT for dot).
      The functions are available in cuda::blas namespace.
     */

    #include <ATen/cuda/CUDAContext.h>

    namespace at {
    namespace cuda {
    namespace blas {

    // RAII guard that sets the CuBLAS pointer mode and restores it to
    // its previous value when the guard is destroyed
    class PointerModeGuard {

      PointerModeGuard(cublasHandle_t handle, cublasPointerMode_t mode) :
          handle(handle) {
        TORCH_CUDABLAS_CHECK(cublasGetPointerMode(handle, &previous_mode));
        TORCH_CUDABLAS_CHECK(cublasSetPointerMode(handle, mode));
      }

      ~PointerModeGuard() {
        cublasSetPointerMode(handle, previous_mode);
      }


      cublasHandle_t handle;
      cublasPointerMode_t previous_mode;
    };

    /* LEVEL 3 BLAS FUNCTIONS */

    #define CUDABLAS_GEMM_ARGTYPES(Dtype)                                       \
      char transa, char transb, i64 m, i64 n, i64 k, Dtype alpha,   \
          const Dtype *a, i64 lda, const Dtype *b, i64 ldb, Dtype beta, \
          Dtype *c, i64 ldc

    template <typename Dtype>
    inline void gemm(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
      AT_ERROR("cuda::blas::gemm: not implemented for ", typeid(Dtype).name());
    }

    template <>
    void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double));
    template <>
    void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float));
    #if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
      template <>
      void gemm<complex<double>>(CUDABLAS_GEMM_ARGTYPES(complex<double>));
    #endif
    #if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
      template <>
      void gemm<complex<float>>(CUDABLAS_GEMM_ARGTYPES(complex<float>));
    #endif
    template <>
    void gemm<Half>(CUDABLAS_GEMM_ARGTYPES(Half));
    #if defined(__HIP_PLATFORM_HCC__) || defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    template <>
    void gemm<BFloat16>(CUDABLAS_GEMM_ARGTYPES(BFloat16));
    #endif

    #define CUDABLAS_BGEMM_ARGTYPES(Dtype)                                       \
      char transa, char transb, i64 m, i64 n, i64 k, Dtype alpha,   \
          const Dtype *a, i64 lda, i64 stridea, \
          const Dtype *b, i64 ldb, i64 strideb, \
          Dtype beta, Dtype *c, i64 ldc, i64 stridec, i64 num_batches

    template <typename Dtype>
    inline void bgemm(CUDABLAS_BGEMM_ARGTYPES(Dtype)) {
      AT_ERROR("cuda::blas::bgemm: not implemented for ", typeid(Dtype).name());
    }

    template <>
    void bgemm<double>(CUDABLAS_BGEMM_ARGTYPES(double));
    template <>
    void bgemm<float>(CUDABLAS_BGEMM_ARGTYPES(float));
    template <>
    void bgemm<complex<double>>(CUDABLAS_BGEMM_ARGTYPES(complex<double>));
    template <>
    void bgemm<complex<float>>(CUDABLAS_BGEMM_ARGTYPES(complex<float>));
    template <>
    void bgemm<Half>(CUDABLAS_BGEMM_ARGTYPES(Half));
    #if defined(__HIP_PLATFORM_HCC__) || defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    template <>
    void bgemm<BFloat16>(CUDABLAS_BGEMM_ARGTYPES(BFloat16));
    #endif

    #define CUDABLAS_TRSM_ARGTYPES(Dtype)                                  \
      cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, \
          cublasOperation_t trans, cublasDiagType_t diag, int m, int n,    \
          const Dtype *alpha, const Dtype *A, int lda, Dtype *B, int ldb

    template <typename Dtype>
    inline void trsm(CUDABLAS_TRSM_ARGTYPES(Dtype)) {
      TORCH_INTERNAL_ASSERT(false, "cuda::blas::trsm: not implemented for ", typeid(Dtype).name());
    }

    template <>
    void trsm<float>(CUDABLAS_TRSM_ARGTYPES(float));
    template <>
    void trsm<double>(CUDABLAS_TRSM_ARGTYPES(double));
    template <>
    void trsm<complex<float>>(CUDABLAS_TRSM_ARGTYPES(complex<float>));
    template <>
    void trsm<complex<double>>(CUDABLAS_TRSM_ARGTYPES(complex<double>));

    #define CUDABLAS_TRSM_BATCHED_ARGTYPES(Dtype)                          \
      cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, \
          cublasOperation_t trans, cublasDiagType_t diag, int m, int n,    \
          const Dtype *alpha, Dtype *A[], int lda, Dtype *B[], int ldb,    \
          int batchCount

    template <typename Dtype>
    inline void trsmBatched(CUDABLAS_TRSM_BATCHED_ARGTYPES(Dtype)) {
      TORCH_INTERNAL_ASSERT(
          false,
          "cuda::blas::trsmBatched: not implemented for ",
          typeid(Dtype).name());
    }

    template <>
    void trsmBatched<float>(CUDABLAS_TRSM_BATCHED_ARGTYPES(float));
    template <>
    void trsmBatched<double>(CUDABLAS_TRSM_BATCHED_ARGTYPES(double));
    template <>
    void trsmBatched<complex<float>>(CUDABLAS_TRSM_BATCHED_ARGTYPES(complex<float>));
    template <>
    void trsmBatched<complex<double>>(CUDABLAS_TRSM_BATCHED_ARGTYPES(complex<double>));

    /* LEVEL 2 BLAS FUNCTIONS */

    #define CUDABLAS_GEMV_ARGTYPES(Dtype)                                         \
      char trans, i64 m, i64 n, Dtype alpha, const Dtype *a, i64 lda, \
          const Dtype *x, i64 incx, Dtype beta, Dtype *y, i64 incy

    template <typename Dtype>
    inline void gemv(CUDABLAS_GEMV_ARGTYPES(Dtype)) {
      AT_ERROR("cuda::blas::gemv: not implemented for ", typeid(Dtype).name());
    }

    template <>
    void gemv<double>(CUDABLAS_GEMV_ARGTYPES(double));
    template <>
    void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float));
    #if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
    template <>
    void gemv<complex<double>>(CUDABLAS_GEMV_ARGTYPES(complex<double>));
    template <>
    void gemv<complex<float>>(CUDABLAS_GEMV_ARGTYPES(complex<float>));
    #endif
    template <>
    void gemv<Half>(CUDABLAS_GEMV_ARGTYPES(Half));
    #if defined(__HIP_PLATFORM_HCC__) || defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    template <>
    void gemv<BFloat16>(CUDABLAS_GEMV_ARGTYPES(BFloat16));
    #endif

    /* LEVEL 1 BLAS FUNCTIONS */

    #define CUDABLAS_DOT_ARGTYPES(Dtype)                                      \
      cublasHandle_t handle, int n, const Dtype *x, int incx, const Dtype *y, \
          int incy, Dtype *result

    template <typename Dtype>
    inline void dot(CUDABLAS_DOT_ARGTYPES(Dtype)) {
      AT_ERROR("cuda::blas::dot: not implemented for ", typeid(Dtype).name());
    }

    template <>
    void dot<double>(CUDABLAS_DOT_ARGTYPES(double));
    template <>
    void dot<float>(CUDABLAS_DOT_ARGTYPES(float));
    template <>
    void dot<Half>(CUDABLAS_DOT_ARGTYPES(Half));
    template <>
    void dot<BFloat16>(CUDABLAS_DOT_ARGTYPES(BFloat16));
    template <>
    void dot<complex<double>>(CUDABLAS_DOT_ARGTYPES(complex<double>));
    template <>
    void dot<complex<float>>(CUDABLAS_DOT_ARGTYPES(complex<float>));

    template <typename Dtype>
    inline void vdot(CUDABLAS_DOT_ARGTYPES(Dtype)) {
      AT_ERROR("cuda::blas::vdot: not implemented for ", typeid(Dtype).name());
    }

    template <>
    void vdot<complex<float>>(CUDABLAS_DOT_ARGTYPES(complex<float>));
    template <>
    void vdot<complex<double>>(CUDABLAS_DOT_ARGTYPES(complex<double>));

    // This guards blocks use of geqrfBatched, getrfBatched, getriBatched on platforms other than cuda
    #ifdef CUDART_VERSION

    #define CUDABLAS_GEQRF_BATCHED_ARGTYPES(Dtype)                   \
      cublasHandle_t handle, int m, int n, Dtype **A_array, int lda, \
          Dtype **tau_array, int *info, int batchsize

    template <class Dtype>
    void geqrfBatched(CUDABLAS_GEQRF_BATCHED_ARGTYPES(Dtype)) {
      TORCH_INTERNAL_ASSERT(
          false,
          "cuda::blas::geqrfBatched: not implemented for ",
          typeid(Dtype).name());
    }
    template <>
    void geqrfBatched<float>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(float));
    template <>
    void geqrfBatched<double>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(double));
    template <>
    void geqrfBatched<complex<double>>(
        CUDABLAS_GEQRF_BATCHED_ARGTYPES(complex<double>));
    template <>
    void geqrfBatched<complex<float>>(
        CUDABLAS_GEQRF_BATCHED_ARGTYPES(complex<float>));

    #define CUDABLAS_GETRF_ARGTYPES(Dtype)  \
      int n, Dtype** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize

    template<class Dtype>
    void getrfBatched(CUDABLAS_GETRF_ARGTYPES(Dtype)) {
      TORCH_CHECK(false, "cuda::blas::getrfBatched: not implemented for ", typeid(Dtype).name());
    }
    template<>
    void getrfBatched<float>(CUDABLAS_GETRF_ARGTYPES(float));
    template<>
    void getrfBatched<double>(CUDABLAS_GETRF_ARGTYPES(double));
    template<>
    void getrfBatched<complex<double>>(CUDABLAS_GETRF_ARGTYPES(complex<double>));
    template<>
    void getrfBatched<complex<float>>(CUDABLAS_GETRF_ARGTYPES(complex<float>));

    #define CUDABLAS_GETRI_ARGTYPES(Dtype)  \
      int n, Dtype** dA_array, int ldda, int* ipiv_array, Dtype** dC_array, int lddc, int* info_array, int batchsize

    template<class Dtype>
    void getriBatched(CUDABLAS_GETRI_ARGTYPES(Dtype)) {
      TORCH_CHECK(false, "cuda::blas::getriBatched: not implemented for ", typeid(Dtype).name());
    }
    template<>
    void getriBatched<float>(CUDABLAS_GETRI_ARGTYPES(float));
    template<>
    void getriBatched<double>(CUDABLAS_GETRI_ARGTYPES(double));
    template<>
    void getriBatched<complex<double>>(CUDABLAS_GETRI_ARGTYPES(complex<double>));
    template<>
    void getriBatched<complex<float>>(CUDABLAS_GETRI_ARGTYPES(complex<float>));

    #define CUDABLAS_GELS_BATCHED_ARGTYPES(Dtype)  \
      cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, Dtype** dA_array, int ldda, Dtype** dC_array, int lddc, int* info, int *devInfoArray, int batchSize

    template <class Dtype>
    void gelsBatched(CUDABLAS_GELS_BATCHED_ARGTYPES(Dtype)) {
      TORCH_INTERNAL_ASSERT(false, "cuda::blas::gelsBatched: not implemented for ", typeid(Dtype).name());
    }

    template<>
    void gelsBatched<double>(CUDABLAS_GELS_BATCHED_ARGTYPES(double));
    template<>
    void gelsBatched<float>(CUDABLAS_GELS_BATCHED_ARGTYPES(float));
    template<>
    void gelsBatched<complex<double>>(CUDABLAS_GELS_BATCHED_ARGTYPES(complex<double>));
    template<>
    void gelsBatched<complex<float>>(CUDABLAS_GELS_BATCHED_ARGTYPES(complex<float>));

    #endif // CUDART_VERSION

    } // namespace blas
    } // namespace cuda
    } // namespace at
    //-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/CUDABlas.cpp]
    /*
      Provides the implementations of CUDA BLAS function templates.
     */

    #include <ATen/cuda/CUDABlas.h>
    #include <ATen/cuda/Exceptions.h>

    #define CUDABLAS_POSINT_CHECK(FD, X)         \
      TORCH_CHECK(                               \
          (X > 0 && X <= INT_MAX),               \
          "cuda::blas::" #FD " argument " #X \
          " must be positive and less than ",    \
          INT_MAX,                               \
          " but got ",                           \
          X)

    #define CUDABLAS_NONNEGINT_CHECK(FD, X)       \
      TORCH_CHECK(                                \
          (X >= 0 && X <= INT_MAX),               \
          "cuda::blas::" #FD " argument " #X  \
          " must be non-negative and less than ", \
          INT_MAX,                                \
          " but got ",                            \
          X)

    namespace {

    static cublasOperation_t _cublasOpFromChar(char op) {
      switch (op) {
        case 'n':
        case 'N':
          return CUBLAS_OP_N;
        case 't':
        case 'T':
          return CUBLAS_OP_T;
        case 'c':
        case 'C':
          return CUBLAS_OP_C;
      }
      AT_ERROR(
          "_cublasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
    }

    static void _cublasAdjustLdLevel2(i64 m, i64 n, i64* lda) {
      // Note: leading dimensions generally are checked that they are > 0
      // and at least as big the result requires (even if the value won't
      // be used).

      // Q: Why does Level3 check trans but this doesn't?
      // A: In level 2, the sizes (m, n) specify the size of A
      // (independent of trans value). In level 3. the sizes (m, n, k)
      // specify the sizes of op(A), op(B) where op depend on trans
      // values.
      if (n <= 1)
        *lda = max<i64>(m, 1);
    }

    static void _cublasAdjustLdLevel3(
        char transa,
        char transb,
        i64 m,
        i64 n,
        i64 k,
        i64* lda,
        i64* ldb,
        i64* ldc) {
      bool transa_ = ((transa == 't') || (transa == 'T'));
      bool transb_ = ((transb == 't') || (transb == 'T'));

      // Note: leading dimensions generally are checked that they are > 0
      // and at least as big the result requires (even if the value won't
      // be used).
      if (n <= 1)
        *ldc = max<i64>(m, 1);

      if (transa_) {
        if (m <= 1)
          *lda = max<i64>(k, 1);
      } else {
        if (k <= 1)
          *lda = max<i64>(m, 1);
      }

      if (transb_) {
        if (k <= 1)
          *ldb = max<i64>(n, 1);
      } else {
        if (n <= 1)
          *ldb = max<i64>(k, 1);
      }
    }
    } // anonymous namespace

    namespace at {
    namespace cuda {
    namespace blas {

    const char* _cublasGetErrorEnum(cublasStatus_t error) {
      if (error == CUBLAS_STATUS_SUCCESS) {
        return "CUBLAS_STATUS_SUCCESS";
      }
      if (error == CUBLAS_STATUS_NOT_INITIALIZED) {
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      }
      if (error == CUBLAS_STATUS_ALLOC_FAILED) {
        return "CUBLAS_STATUS_ALLOC_FAILED";
      }
      if (error == CUBLAS_STATUS_INVALID_VALUE) {
        return "CUBLAS_STATUS_INVALID_VALUE";
      }
      if (error == CUBLAS_STATUS_ARCH_MISMATCH) {
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      }
      if (error == CUBLAS_STATUS_MAPPING_ERROR) {
        return "CUBLAS_STATUS_MAPPING_ERROR";
      }
      if (error == CUBLAS_STATUS_EXECUTION_FAILED) {
        return "CUBLAS_STATUS_EXECUTION_FAILED";
      }
      if (error == CUBLAS_STATUS_INTERNAL_ERROR) {
        return "CUBLAS_STATUS_INTERNAL_ERROR";
      }
      if (error == CUBLAS_STATUS_NOT_SUPPORTED) {
        return "CUBLAS_STATUS_NOT_SUPPORTED";
      }
    #ifdef CUBLAS_STATUS_LICENSE_ERROR
      if (error == CUBLAS_STATUS_LICENSE_ERROR) {
        return "CUBLAS_STATUS_LICENSE_ERROR";
      }
    #endif
      return "<unknown>";
    }

    /* LEVEL 3 BLAS FUNCTIONS */

    #ifndef __HIP_PLATFORM_HCC__
    #if defined(CUDA_VERSION) && CUDA_VERSION >= 11200
    #define cublasGemmStridedBatchedExFix cublasGemmStridedBatchedEx
    #else
    // Workaround for https://github.com/pytorch/pytorch/issues/45724
    cublasStatus_t cublasGemmStridedBatchedExFix(cublasHandle_t &handle,
      cublasOperation_t transa,
      cublasOperation_t transb,
      int m,
      int n,
      int k,
      const void    *alpha,
      const void     *A,
      cudaDataType Atype,
      int lda,
      long long int strideA,
      const void     *B,
      cudaDataType Btype,
      int ldb,
      long long int strideB,
      const void    *beta,
      void           *C,
      cudaDataType Ctype,
      int ldc,
      long long int strideC,
      i64 batchCount,
      cudaDataType computeType,
      cublasGemmAlgo_t algo)
    {
      cudaDeviceProp* prop = cuda::getCurrentDeviceProperties();
      if (prop->major != 7) {
        return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
      }
      cublasStatus_t result;
      constexpr i64 split = 63 * 1024;
      for(i64 i = 0; i < batchCount; i += split) {
        i64 count = min<i64>(split, batchCount - i);
        result = cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha,
          (char *)A + i * strideA * 2, Atype, lda, strideA,
          (char *)B + i * strideB * 2, Btype, ldb, strideB,
          beta,
          (char *)C + i * strideC * 2, Ctype, ldc, strideC,
          (int)count, computeType, algo);
        TORCH_CUDABLAS_CHECK(result);
      }
      return result;
    }
    #endif
    #endif

    #define GEMM_CHECK_ARGVALUES(Dtype)           \
      do {                                        \
        CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, m); \
        CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, n); \
        CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, k); \
        CUDABLAS_POSINT_CHECK(gemm<Dtype>, lda);  \
        CUDABLAS_POSINT_CHECK(gemm<Dtype>, ldb);  \
        CUDABLAS_POSINT_CHECK(gemm<Dtype>, ldc);  \
      } while (0)

    #define BGEMM_CHECK_ARGVALUES(Dtype)           \
      do {                                        \
        CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, m); \
        CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, n); \
        CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, k); \
        CUDABLAS_POSINT_CHECK(bgemm<Dtype>, lda);  \
        CUDABLAS_POSINT_CHECK(bgemm<Dtype>, ldb);  \
        CUDABLAS_POSINT_CHECK(bgemm<Dtype>, ldc);  \
        CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, num_batches);  \
      } while (0)

    template <>
    void bgemm<double>(CUDABLAS_BGEMM_ARGTYPES(double)) {
      // See Note [Writing Nondeterministic Operations]
      globalContext().alertCuBLASConfigNotDeterministic();
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t opa = _cublasOpFromChar(transa);
      cublasOperation_t opb = _cublasOpFromChar(transb);
      _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
      BGEMM_CHECK_ARGVALUES(double);
      TORCH_CUDABLAS_CHECK(cublasDgemmStridedBatched(
          handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
    }

    template <>
    void bgemm<float>(CUDABLAS_BGEMM_ARGTYPES(float)) {
      // See Note [Writing Nondeterministic Operations]
      globalContext().alertCuBLASConfigNotDeterministic();
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t opa = _cublasOpFromChar(transa);
      cublasOperation_t opb = _cublasOpFromChar(transb);
      _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
      BGEMM_CHECK_ARGVALUES(float);
      TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
          handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
    }

    template <>
    void bgemm<complex<double>>(CUDABLAS_BGEMM_ARGTYPES(complex<double>)) {
      // See Note [Writing Nondeterministic Operations]
      globalContext().alertCuBLASConfigNotDeterministic();
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t opa = _cublasOpFromChar(transa);
      cublasOperation_t opb = _cublasOpFromChar(transb);
      _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
      BGEMM_CHECK_ARGVALUES(complex<double>);
      TORCH_CUDABLAS_CHECK(cublasZgemmStridedBatched(
          handle, opa, opb, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
          lda, stridea, reinterpret_cast<const cuDoubleComplex*>(b), ldb, strideb, reinterpret_cast<const cuDoubleComplex*>(&beta),
          reinterpret_cast<cuDoubleComplex*>(c), ldc, stridec, num_batches));
    }

    template <>
    void bgemm<complex<float>>(CUDABLAS_BGEMM_ARGTYPES(complex<float>)) {
      // See Note [Writing Nondeterministic Operations]
      globalContext().alertCuBLASConfigNotDeterministic();
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t opa = _cublasOpFromChar(transa);
      cublasOperation_t opb = _cublasOpFromChar(transb);
      _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
      BGEMM_CHECK_ARGVALUES(complex<float>);
      TORCH_CUDABLAS_CHECK(cublasCgemmStridedBatched(
          handle, opa, opb, m, n, k, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
          lda, stridea, reinterpret_cast<const cuComplex*>(b), ldb, strideb, reinterpret_cast<const cuComplex*>(&beta),
          reinterpret_cast<cuComplex*>(c), ldc, stridec, num_batches));
    }

    template <>
    void bgemm<Half>(CUDABLAS_BGEMM_ARGTYPES(Half)) {
      // See Note [Writing Nondeterministic Operations]
      globalContext().alertCuBLASConfigNotDeterministic();
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t opa = _cublasOpFromChar(transa);
      cublasOperation_t opb = _cublasOpFromChar(transb);
      _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
      BGEMM_CHECK_ARGVALUES(Half);
      float falpha = alpha;
      float fbeta = beta;
    #ifdef __HIP_PLATFORM_HCC__
      TORCH_CUDABLAS_CHECK(rocblas_gemm_strided_batched_ex(handle, opa, opb, (int)m, (int)n, (int)k,
                                       (void*)&falpha, a, rocblas_datatype_f16_r, (int)lda, stridea,
                                       b, rocblas_datatype_f16_r, (int)ldb, strideb,
                                       (void*)&fbeta, c, rocblas_datatype_f16_r, (int)ldc, stridec,
                                       c, rocblas_datatype_f16_r, (int)ldc, stridec,
                                       (int) num_batches, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
                                       0, 0));
    #else
      #if defined(CUDA_VERSION) && CUDA_VERSION < 11000
        // On CUDA versions prior to 11, users are required to set the math mode to CUBLAS_TENSOR_OP_MATH
        // manually to be able to use tensor cores for FP16. On CUDA 11, this is no longer required.
        TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
      #endif  // CUDA_VERSION < 11000

      cudaDeviceProp* prop = cuda::getCurrentDeviceProperties();
      if (prop->major >= 5){
        TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedExFix(
          handle, opa, opb, m, n, k,
          (void*)(&falpha), a, CUDA_R_16F, lda, stridea,
          b, CUDA_R_16F, ldb, strideb, (void*)(&fbeta),
          c, CUDA_R_16F, ldc, stridec,
          num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      } else {
        for (i64 i = 0; i < num_batches; ++i) {
          cuda::blas::gemm<Half>(
            transa, transb,
            m, n, k,
            alpha, (a + i * stridea), lda,
            (b + i * strideb), ldb, beta,
            (c + i * stridec), ldc);
        }
      }
      #if defined(CUDA_VERSION) && CUDA_VERSION < 11000
        // On CUDA versions prior to 11, users are required to set the math mode to CUBLAS_TENSOR_OP_MATH
        // manually to be able to use tensor cores for FP16. On CUDA 11, this is no longer required.
        TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
      #endif  // CUDA_VERSION < 11000
    #endif // __HIP_PLATFORM_HCC__
    }

    #if defined(__HIP_PLATFORM_HCC__) || defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    template <>
    void bgemm<BFloat16>(CUDABLAS_BGEMM_ARGTYPES(BFloat16)) {
      // See Note [Writing Nondeterministic Operations]
      globalContext().alertCuBLASConfigNotDeterministic();
      BGEMM_CHECK_ARGVALUES(BFloat16);
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t opa = _cublasOpFromChar(transa);
      cublasOperation_t opb = _cublasOpFromChar(transb);
      const float falpha = alpha;
      const float fbeta = beta;
      _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

      #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
        TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedExFix(handle,
                                        opa, opb, (int)m, (int)n, (int)k,
                                        (void*)&falpha, a, CUDA_R_16BF, (int)lda, stridea,
                                        b, CUDA_R_16BF, (int)ldb, strideb,
                                        (void*)&fbeta, c, CUDA_R_16BF, (int)ldc, stridec,
                                        (int)num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      #elif defined(__HIP_PLATFORM_HCC__)
        TORCH_CUDABLAS_CHECK(rocblas_gemm_strided_batched_ex(handle, opa, opb, (int)m, (int)n, (int)k,
                                       (void*)&falpha, a, rocblas_datatype_bf16_r, (int)lda, stridea,
                                       b, rocblas_datatype_bf16_r, (int)ldb, strideb,
                                       (void*)&fbeta, c, rocblas_datatype_bf16_r, (int)ldc, stridec,
                                       c, rocblas_datatype_bf16_r, (int)ldc, stridec,
                                       (int) num_batches, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
                                       0, 0, NULL, NULL));
      #else
        TORCH_CHECK(false, "CUDA BFloat16 bgemm requires CUDA 11 or later");
      #endif // defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    }
    #endif // __HIP_PLATFORM_HCC__

    template <>
    void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
      // See Note [Writing Nondeterministic Operations]
      globalContext().alertCuBLASConfigNotDeterministic();
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t opa = _cublasOpFromChar(transa);
      cublasOperation_t opb = _cublasOpFromChar(transb);
      _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
      GEMM_CHECK_ARGVALUES(double);
      TORCH_CUDABLAS_CHECK(cublasDgemm(
          handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
    }

    template <>
    void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
      // See Note [Writing Nondeterministic Operations]
      globalContext().alertCuBLASConfigNotDeterministic();
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t opa = _cublasOpFromChar(transa);
      cublasOperation_t opb = _cublasOpFromChar(transb);
      _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
      GEMM_CHECK_ARGVALUES(float);
      TORCH_CUDABLAS_CHECK(cublasSgemm(
          handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
    }

    #if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
      template <>
      void gemm<complex<double>>(CUDABLAS_GEMM_ARGTYPES(complex<double>)) {
        // See Note [Writing Nondeterministic Operations]
        globalContext().alertCuBLASConfigNotDeterministic();
        cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
        cublasOperation_t opa = _cublasOpFromChar(transa);
        cublasOperation_t opb = _cublasOpFromChar(transb);
        _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
        GEMM_CHECK_ARGVALUES(complex<double>);
        TORCH_CUDABLAS_CHECK(cublasZgemm(
            handle, opa, opb, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
            lda, reinterpret_cast<const cuDoubleComplex*>(b), ldb, reinterpret_cast<const cuDoubleComplex*>(&beta),
            reinterpret_cast<cuDoubleComplex*>(c), ldc));
      }
    #endif

    #if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
      template <>
      void gemm<complex<float>>(CUDABLAS_GEMM_ARGTYPES(complex<float>)) {
        // See Note [Writing Nondeterministic Operations]
        globalContext().alertCuBLASConfigNotDeterministic();
        cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
        cublasOperation_t opa = _cublasOpFromChar(transa);
        cublasOperation_t opb = _cublasOpFromChar(transb);
        _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
        GEMM_CHECK_ARGVALUES(complex<float>);
        TORCH_CUDABLAS_CHECK(cublasCgemm(
            handle, opa, opb, m, n, k, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
            lda, reinterpret_cast<const cuComplex*>(b), ldb, reinterpret_cast<const cuComplex*>(&beta),
            reinterpret_cast<cuComplex*>(c), ldc));
      }
    #endif

    template <>
    void gemm<Half>(CUDABLAS_GEMM_ARGTYPES(Half)) {
      // See Note [Writing Nondeterministic Operations]
      globalContext().alertCuBLASConfigNotDeterministic();
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t opa = _cublasOpFromChar(transa);
      cublasOperation_t opb = _cublasOpFromChar(transb);
      float falpha = alpha;
      float fbeta = beta;
      _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
      GEMM_CHECK_ARGVALUES(Half);
    #ifdef __HIP_PLATFORM_HCC__
      TORCH_CUDABLAS_CHECK(rocblas_gemm_ex(
          handle,
          opa,
          opb,
          m,
          n,
          k,
          &falpha,
          a,
          rocblas_datatype_f16_r,
          lda,
          b,
          rocblas_datatype_f16_r,
          ldb,
          &fbeta,
          c,
          rocblas_datatype_f16_r,
          ldc,
          c,
          rocblas_datatype_f16_r,
          ldc,
          rocblas_datatype_f32_r,
          rocblas_gemm_algo_standard,
          0,
          0));
    #else
      cudaDeviceProp* prop = cuda::getCurrentDeviceProperties();
      if (prop->major >= 5) {
    #if defined(CUDA_VERSION) && CUDA_VERSION < 11000
        // On CUDA versions prior to 11, users are required to set the math mode to CUBLAS_TENSOR_OP_MATH
        // manually to be able to use tensor cores for FP16. On CUDA 11, this is no longer required.
        TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    #endif  // CUDA_VERSION < 11000
        TORCH_CUDABLAS_CHECK(cublasGemmEx(
            handle,
            opa,
            opb,
            m,
            n,
            k,
            &falpha,
            a,
            CUDA_R_16F,
            lda,
            b,
            CUDA_R_16F,
            ldb,
            &fbeta,
            c,
            CUDA_R_16F,
            ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DFALT_TENSOR_OP));
    #if defined(CUDA_VERSION) && CUDA_VERSION < 11000
        // On CUDA versions prior to 11, users are required to set the math mode to CUBLAS_TENSOR_OP_MATH
        // manually to be able to use tensor cores for FP16. On CUDA 11, this is no longer required.
        TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    #endif  // CUDA_VERSION < 11000
      } else {
        TORCH_CUDABLAS_CHECK(cublasSgemmEx(
            handle,
            opa,
            opb,
            m,
            n,
            k,
            &falpha,
            a,
            CUDA_R_16F,
            lda,
            b,
            CUDA_R_16F,
            ldb,
            &fbeta,
            c,
            CUDA_R_16F,
            ldc));
      }
    #endif
    }

    #ifdef __HIP_PLATFORM_HCC__
    template <>
    void gemm<BFloat16>(CUDABLAS_GEMM_ARGTYPES(BFloat16)) {
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t opa = _cublasOpFromChar(transa);
      cublasOperation_t opb = _cublasOpFromChar(transb);
      float falpha = alpha;
      float fbeta = beta;
      _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
      GEMM_CHECK_ARGVALUES(BFloat16);
      TORCH_CUDABLAS_CHECK(rocblas_gemm_ex(
          handle,
          opa,
          opb,
          m,
          n,
          k,
          &falpha,
          a,
          rocblas_datatype_bf16_r,
          lda,
          b,
          rocblas_datatype_bf16_r,
          ldb,
          &fbeta,
          c,
          rocblas_datatype_bf16_r,
          ldc,
          c,
          rocblas_datatype_bf16_r,
          ldc,
          rocblas_datatype_f32_r,
          rocblas_gemm_algo_standard,
          0,
          0));
    }
    #endif

    #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    template <>
    void gemm<BFloat16>(CUDABLAS_GEMM_ARGTYPES(BFloat16)) {
      globalContext().alertCuBLASConfigNotDeterministic();
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t opa = _cublasOpFromChar(transa);
      cublasOperation_t opb = _cublasOpFromChar(transb);
      float falpha = alpha;
      float fbeta = beta;
      _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
      GEMM_CHECK_ARGVALUES(BFloat16);
      TORCH_CUDABLAS_CHECK(cublasGemmEx(
          handle,
          opa,
          opb,
          m,
          n,
          k,
          &falpha,
          a,
          CUDA_R_16BF,
          lda,
          b,
          CUDA_R_16BF,
          ldb,
          &fbeta,
          c,
          CUDA_R_16BF,
          ldc,
          CUDA_R_32F,
          CUBLAS_GEMM_DFALT_TENSOR_OP));
    }
    #endif

    template <>
    void trsm<float>(CUDABLAS_TRSM_ARGTYPES(float)) {
      TORCH_CUDABLAS_CHECK(cublasStrsm(
          handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }

    template <>
    void trsm<double>(CUDABLAS_TRSM_ARGTYPES(double)) {
      TORCH_CUDABLAS_CHECK(cublasDtrsm(
          handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    }

    template <>
    void trsm<complex<float>>(CUDABLAS_TRSM_ARGTYPES(complex<float>)) {
      TORCH_CUDABLAS_CHECK(cublasCtrsm(
          handle,
          side,
          uplo,
          trans,
          diag,
          m,
          n,
          reinterpret_cast<const cuComplex*>(alpha),
          reinterpret_cast<const cuComplex*>(A),
          lda,
          reinterpret_cast<cuComplex*>(B),
          ldb));
    }

    template <>
    void trsm<complex<double>>(CUDABLAS_TRSM_ARGTYPES(complex<double>)) {
      TORCH_CUDABLAS_CHECK(cublasZtrsm(
          handle,
          side,
          uplo,
          trans,
          diag,
          m,
          n,
          reinterpret_cast<const cuDoubleComplex*>(alpha),
          reinterpret_cast<const cuDoubleComplex*>(A),
          lda,
          reinterpret_cast<cuDoubleComplex*>(B),
          ldb));
    }

    template <>
    void trsmBatched<float>(CUDABLAS_TRSM_BATCHED_ARGTYPES(float)) {
      TORCH_CUDABLAS_CHECK(cublasStrsmBatched(
          handle,
          side,
          uplo,
          trans,
          diag,
          m,
          n,
          alpha,
          A,
          lda,
          B,
          ldb,
          batchCount));
    }

    template <>
    void trsmBatched<double>(CUDABLAS_TRSM_BATCHED_ARGTYPES(double)) {
      TORCH_CUDABLAS_CHECK(cublasDtrsmBatched(
          handle,
          side,
          uplo,
          trans,
          diag,
          m,
          n,
          alpha,
          A,
          lda,
          B,
          ldb,
          batchCount));
    }

    template <>
    void trsmBatched<complex<float>>(
        CUDABLAS_TRSM_BATCHED_ARGTYPES(complex<float>)) {
      TORCH_CUDABLAS_CHECK(cublasCtrsmBatched(
          handle,
          side,
          uplo,
          trans,
          diag,
          m,
          n,
          reinterpret_cast<const cuComplex*>(alpha),
          reinterpret_cast<cuComplex**>(A),
          lda,
          reinterpret_cast<cuComplex**>(B),
          ldb,
          batchCount));
    }

    template <>
    void trsmBatched<complex<double>>(
        CUDABLAS_TRSM_BATCHED_ARGTYPES(complex<double>)) {
      TORCH_CUDABLAS_CHECK(cublasZtrsmBatched(
          handle,
          side,
          uplo,
          trans,
          diag,
          m,
          n,
          reinterpret_cast<const cuDoubleComplex*>(alpha),
          reinterpret_cast<cuDoubleComplex**>(A),
          lda,
          reinterpret_cast<cuDoubleComplex**>(B),
          ldb,
          batchCount));
    }

    /* LEVEL 2 BLAS FUNCTIONS */

    #define GEMV_CHECK_ARGVALUES(Dtype)           \
      do {                                        \
        CUDABLAS_NONNEGINT_CHECK(gemv<Dtype>, m); \
        CUDABLAS_NONNEGINT_CHECK(gemv<Dtype>, n); \
        CUDABLAS_POSINT_CHECK(gemv<Dtype>, lda);  \
        CUDABLAS_POSINT_CHECK(gemv<Dtype>, incx); \
        CUDABLAS_POSINT_CHECK(gemv<Dtype>, incy); \
      } while (0)

    #if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
      template <>
      void gemv<complex<double>>(CUDABLAS_GEMV_ARGTYPES(complex<double>)) {
        // See Note [Writing Nondeterministic Operations]
        globalContext().alertCuBLASConfigNotDeterministic();
        cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
        cublasOperation_t op = _cublasOpFromChar(trans);
        _cublasAdjustLdLevel2(m, n, &lda);
        GEMV_CHECK_ARGVALUES(complex<double>);
        TORCH_CUDABLAS_CHECK(
            cublasZgemv(handle, op, m, n, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
            lda, reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<const cuDoubleComplex*>(&beta),
            reinterpret_cast<cuDoubleComplex*>(y), incy));
      }
    #endif

    #if !defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 210)
    template <>
    void gemv<complex<float>>(CUDABLAS_GEMV_ARGTYPES(complex<float>)) {
      // gemv is bw bound, and does not benefit from TF32. But the precision
      // loss still happens on TF32. So we disable it here.
      NoTF32Guard disable_tf32;
      // See Note [Writing Nondeterministic Operations]
      globalContext().alertCuBLASConfigNotDeterministic();
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t op = _cublasOpFromChar(trans);
      _cublasAdjustLdLevel2(m, n, &lda);
      GEMV_CHECK_ARGVALUES(complex<float>);
      TORCH_CUDABLAS_CHECK(
          cublasCgemv(handle, op, m, n, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
          lda, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<const cuComplex*>(&beta),
          reinterpret_cast<cuComplex*>(y), incy));
    }
    #endif

    template <>
    void gemv<double>(CUDABLAS_GEMV_ARGTYPES(double)) {
      // See Note [Writing Nondeterministic Operations]
      globalContext().alertCuBLASConfigNotDeterministic();
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t op = _cublasOpFromChar(trans);
      _cublasAdjustLdLevel2(m, n, &lda);
      GEMV_CHECK_ARGVALUES(double);
      TORCH_CUDABLAS_CHECK(
          cublasDgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
    }

    template <>
    void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float)) {
      // gemv is bw bound, and does not benefit from TF32. But the precision
      // loss still happens on TF32. So we disable it here.
      NoTF32Guard disable_tf32;
      // See Note [Writing Nondeterministic Operations]
      globalContext().alertCuBLASConfigNotDeterministic();
      cublasHandle_t handle = cuda::getCurrentCUDABlasHandle();
      cublasOperation_t op = _cublasOpFromChar(trans);
      _cublasAdjustLdLevel2(m, n, &lda);
      GEMV_CHECK_ARGVALUES(float);
      TORCH_CUDABLAS_CHECK(
          cublasSgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
    }

    template <>
    void gemv<Half>(CUDABLAS_GEMV_ARGTYPES(Half)) {
      // In general, cublas regards matrices as column-major.
      // The cublasS/Dgemv usages in cuda::blas::gemv<float>/<double> above
      // require that external blas::gemv callers obey the following convention:
      //
      // If "a" is row-major with shape (output, summed) in blas::gemv's caller,
      // caller interprets it as column-major with shape (summed, output), passes
      // summed and output respectively to our local vars m, n, and requests that cublas
      // internally transpose ("trans") the column-major interpretation of a.
      //
      // There's no such thing as "cublasHalfgemv", so here we hack gemv with a gemm.
      // However, we must allow the same calling convention, because the caller shouldn't
      // have to swap args based on whether it's calling blas::gemv<Half> or <float>.

      bool trans_bool = (_cublasOpFromChar(trans) != CUBLAS_OP_N);
      if (trans_bool) {
        swap(m, n);
      }
      // After swap, local vars m, n contain the output and summed sizes respectively,
      // regardless of whether "a" was row-major or column-major in gemv<>'s caller.

      // To handle the possibility incy > 1, interprets vector y as column-major matrix with one row
      // (shape (1, output)) and leading dim incy.
      // trans(a)*x would compute a matrix with one column (shape (output, 1)) which wouldn't match y.
      // So instead, we interpret x similarly to y, as a column-major matrix with one row
      // (shape (1, summed)) and leading dim incx.  The gemm then carries out x*transpose(trans(a)) to
      // produce a matrix with one row (shape (1, output)), matching y.
      char trans_flipped = (trans_bool ? 'n' : 't');
      gemm<Half>(
          'n', trans_flipped, 1, m, n, alpha, x, incx, a, lda, beta, y, incy);
    }

    #if defined(__HIP_PLATFORM_HCC__) || defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    template <>
    void gemv<BFloat16>(CUDABLAS_GEMV_ARGTYPES(BFloat16)) {
      bool trans_bool = (_cublasOpFromChar(trans) != CUBLAS_OP_N);
      if (trans_bool) {
        swap(m, n);
      }
      char trans_flipped = (trans_bool ? 'n' : 't');
      gemm<BFloat16>(
          'n', trans_flipped, 1, m, n, alpha, x, incx, a, lda, beta, y, incy);
    }
    #endif

    /* LEVEL 1 BLAS FUNCTIONS */

    template <>
    void dot<double>(CUDABLAS_DOT_ARGTYPES(double)) {
      TORCH_CUDABLAS_CHECK(cublasDdot(handle, n, x, incx, y, incy, result));
    }

    template <>
    void dot<float>(CUDABLAS_DOT_ARGTYPES(float)) {
      TORCH_CUDABLAS_CHECK(cublasSdot(handle, n, x, incx, y, incy, result));
    }

    template <>
    void dot<complex<double>>(CUDABLAS_DOT_ARGTYPES(complex<double>)) {
      TORCH_CUDABLAS_CHECK(cublasZdotu(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                                       incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                                       reinterpret_cast<cuDoubleComplex*>(result)));
    }

    template <>
    void dot<complex<float>>(CUDABLAS_DOT_ARGTYPES(complex<float>)) {
      TORCH_CUDABLAS_CHECK(cublasCdotu(handle, n, reinterpret_cast<const cuComplex*>(x),
                                       incx, reinterpret_cast<const cuComplex*>(y), incy,
                                       reinterpret_cast<cuComplex*>(result)));
    }

    template <>
    void dot<Half>(CUDABLAS_DOT_ARGTYPES(Half)) {
    #if CUDA_VERSION >= 8000
      TORCH_CUDABLAS_CHECK(cublasDotEx(
          handle,
          n,
          x,
          CUDA_R_16F,
          incx,
          y,
          CUDA_R_16F,
          incy,
          result,
          CUDA_R_16F,
          CUDA_R_32F));
    #elif HIP_VERSION >= 210
      TORCH_CUDABLAS_CHECK(rocblas_hdot(
          handle,
          n,
          reinterpret_cast<const rocblas_half*>(x),
          incx,
          reinterpret_cast<const rocblas_half*>(y),
          incy,
          reinterpret_cast<rocblas_half*>(result)));
    #else
      AT_ERROR("Cublas_Hdot requires CUDA 8.0+");
    #endif
    }

    template <>
    void dot<BFloat16>(CUDABLAS_DOT_ARGTYPES(BFloat16)) {
    #if CUDA_VERSION >= 11000
      TORCH_CUDABLAS_CHECK(cublasDotEx(
          handle,
          n,
          x,
          CUDA_R_16BF,
          incx,
          y,
          CUDA_R_16BF,
          incy,
          result,
          CUDA_R_16BF,
          CUDA_R_32F));
    #elif HIP_VERSION >= 210
      TORCH_CUDABLAS_CHECK(rocblas_bfdot(
          handle,
          n,
          reinterpret_cast<const rocblas_bfloat16*>(x),
          incx,
          reinterpret_cast<const rocblas_bfloat16*>(y),
          incy,
          reinterpret_cast<rocblas_bfloat16*>(result)));
    #else
      AT_ERROR("Cublas_bfdot requires CUDA 11.0+");
    #endif
    }

    template <>
    void vdot<complex<float>>(CUDABLAS_DOT_ARGTYPES(complex<float>)) {
      TORCH_CUDABLAS_CHECK(cublasCdotc(handle, n, reinterpret_cast<const cuComplex*>(x),
                                       incx, reinterpret_cast<const cuComplex*>(y), incy,
                                       reinterpret_cast<cuComplex*>(result)));
    }

    template <>
    void vdot<complex<double>>(CUDABLAS_DOT_ARGTYPES(complex<double>)) {
      TORCH_CUDABLAS_CHECK(cublasZdotc(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                                       incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                                       reinterpret_cast<cuDoubleComplex*>(result)));
    }

    // This guards blocks use of geqrfBatched, getrfBatched, getriBatched on platforms other than cuda
    #ifdef CUDART_VERSION

    template <>
    void geqrfBatched<float>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(float)) {
      TORCH_CUDABLAS_CHECK(cublasSgeqrfBatched(
          handle, m, n, A_array, lda, tau_array, info, batchsize));
    }

    template <>
    void geqrfBatched<double>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(double)) {
      TORCH_CUDABLAS_CHECK(cublasDgeqrfBatched(
          handle, m, n, A_array, lda, tau_array, info, batchsize));
    }

    template <>
    void geqrfBatched<complex<float>>(
        CUDABLAS_GEQRF_BATCHED_ARGTYPES(complex<float>)) {
      TORCH_CUDABLAS_CHECK(cublasCgeqrfBatched(
          handle,
          m,
          n,
          reinterpret_cast<cuComplex**>(A_array),
          lda,
          reinterpret_cast<cuComplex**>(tau_array),
          info,
          batchsize));
    }

    template <>
    void geqrfBatched<complex<double>>(
        CUDABLAS_GEQRF_BATCHED_ARGTYPES(complex<double>)) {
      TORCH_CUDABLAS_CHECK(cublasZgeqrfBatched(
          handle,
          m,
          n,
          reinterpret_cast<cuDoubleComplex**>(A_array),
          lda,
          reinterpret_cast<cuDoubleComplex**>(tau_array),
          info,
          batchsize));
    }

    template <>
    void getrfBatched<double>(
        int n, double** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize) {
      auto handle = cuda::getCurrentCUDABlasHandle();
      TORCH_CUDABLAS_CHECK(cublasDgetrfBatched(
          handle, n, dA_array, ldda, ipiv_array, info_array, batchsize));
    }

    template <>
    void getrfBatched<float>(
        int n, float** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize) {
      auto handle = cuda::getCurrentCUDABlasHandle();
      TORCH_CUDABLAS_CHECK(cublasSgetrfBatched(
          handle, n, dA_array, ldda, ipiv_array, info_array, batchsize));
    }

    template <>
    void getrfBatched<complex<double>>(
        int n,
        complex<double>** dA_array,
        int ldda,
        int* ipiv_array,
        int* info_array,
        int batchsize) {
      auto handle = cuda::getCurrentCUDABlasHandle();
      TORCH_CUDABLAS_CHECK(cublasZgetrfBatched(
          handle,
          n,
          reinterpret_cast<cuDoubleComplex**>(dA_array),
          ldda,
          ipiv_array,
          info_array,
          batchsize));
    }

    template <>
    void getrfBatched<complex<float>>(
        int n,
        complex<float>** dA_array,
        int ldda,
        int* ipiv_array,
        int* info_array,
        int batchsize) {
      auto handle = cuda::getCurrentCUDABlasHandle();
      TORCH_CUDABLAS_CHECK(cublasCgetrfBatched(
          handle,
          n,
          reinterpret_cast<cuComplex**>(dA_array),
          ldda,
          ipiv_array,
          info_array,
          batchsize));
    }

    template <>
    void getriBatched<double>(
        int n, double** dA_array, int ldda, int* ipiv_array, double** dC_array, int lddc, int* info_array, int batchsize) {
      auto handle = cuda::getCurrentCUDABlasHandle();
      TORCH_CUDABLAS_CHECK(cublasDgetriBatched(
          handle, n, dA_array, ldda, ipiv_array, dC_array, lddc, info_array, batchsize));
    }

    template <>
    void getriBatched<float>(
        int n, float** dA_array, int ldda, int* ipiv_array, float** dC_array, int lddc, int* info_array, int batchsize) {
      auto handle = cuda::getCurrentCUDABlasHandle();
      TORCH_CUDABLAS_CHECK(cublasSgetriBatched(
          handle, n, dA_array, ldda, ipiv_array, dC_array, lddc, info_array, batchsize));
    }

    template <>
    void getriBatched<complex<double>>(
        int n,
        complex<double>** dA_array,
        int ldda,
        int* ipiv_array,
        complex<double>** dC_array,
        int lddc,
        int* info_array,
        int batchsize) {
      auto handle = cuda::getCurrentCUDABlasHandle();
      TORCH_CUDABLAS_CHECK(cublasZgetriBatched(
          handle,
          n,
          reinterpret_cast<cuDoubleComplex**>(dA_array),
          ldda,
          ipiv_array,
          reinterpret_cast<cuDoubleComplex**>(dC_array),
          lddc,
          info_array,
          batchsize));
    }

    template <>
    void getriBatched<complex<float>>(
        int n,
        complex<float>** dA_array,
        int ldda,
        int* ipiv_array,
        complex<float>** dC_array,
        int lddc,
        int* info_array,
        int batchsize) {
      auto handle = cuda::getCurrentCUDABlasHandle();
      TORCH_CUDABLAS_CHECK(cublasCgetriBatched(
          handle,
          n,
          reinterpret_cast<cuComplex**>(dA_array),
          ldda,
          ipiv_array,
          reinterpret_cast<cuComplex**>(dC_array),
          lddc,
          info_array,
          batchsize));
    }

    template <>
    void gelsBatched<double>(CUDABLAS_GELS_BATCHED_ARGTYPES(double)) {
      TORCH_CUDABLAS_CHECK(cublasDgelsBatched(
          handle, trans, m, n, nrhs, dA_array, ldda, dC_array, lddc, info, devInfoArray, batchSize));
    }

    template <>
    void gelsBatched<float>(CUDABLAS_GELS_BATCHED_ARGTYPES(float)) {
      TORCH_CUDABLAS_CHECK(cublasSgelsBatched(
          handle, trans, m, n, nrhs, dA_array, ldda, dC_array, lddc, info, devInfoArray, batchSize));
    }

    template <>
    void gelsBatched<complex<double>>(CUDABLAS_GELS_BATCHED_ARGTYPES(complex<double>)) {
      TORCH_CUDABLAS_CHECK(cublasZgelsBatched(
          handle, trans,
          m, n, nrhs,
          reinterpret_cast<cuDoubleComplex**>(dA_array),
          ldda,
          reinterpret_cast<cuDoubleComplex**>(dC_array),
          lddc,
          info,
          devInfoArray,
          batchSize));
    }

    template <>
    void gelsBatched<complex<float>>(CUDABLAS_GELS_BATCHED_ARGTYPES(complex<float>)) {
      TORCH_CUDABLAS_CHECK(cublasCgelsBatched(
          handle, trans,
          m, n, nrhs,
          reinterpret_cast<cuComplex**>(dA_array),
          ldda,
          reinterpret_cast<cuComplex**>(dC_array),
          lddc,
          info,
          devInfoArray,
          batchSize));
    }

    #endif // CUDART_VERSION

    } // namespace blas
    } // namespace cuda
    } // namespace at
    */
}

