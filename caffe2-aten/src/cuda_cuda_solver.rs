crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/CUDASolver.h]

/// cuSOLVER version >= 11000 includes 64-bit API
#[cfg(all(CUDART_VERSION,CUSOLVER_VERSION,CUSOLVER_VERSION_GTE_11000))]
pub const USE_CUSOLVER_64_BIT: bool = true;

#[cfg(CUDART_VERSION)]
#[macro_export] macro_rules! cudasolver_getrf_argtypes {
    ($Dtype:ident) => {
        /*
        
            cusolverDnHandle_t handle, int m, int n, Dtype* dA, int ldda, int* ipiv, int* info
        */
    }
}

#[cfg(CUDART_VERSION)]
pub fn getrf<Dtype>(
        handle: CuSolverDnHandle,
        m:      i32,
        n:      i32,
        da:     *mut Dtype,
        ldda:   i32,
        ipiv:   *mut i32,
        info:   *mut i32)  {

    todo!();
        /*
            TORCH_CHECK(false, "at::cuda::solver::getrf: not implemented for ", typeid(Dtype).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn getrs<Dtype>(
        handle: CuSolverDnHandle,
        n:      i32,
        nrhs:   i32,
        da:     *mut Dtype,
        lda:    i32,
        ipiv:   *mut i32,
        ret:    *mut Dtype,
        ldb:    i32,
        info:   *mut i32)  {

    todo!();
        /*
            TORCH_CHECK(false, "at::cuda::solver::getrs: not implemented for ", typeid(Dtype).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn gesvdj<Dtype, Vtype>(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        econ:   i32,
        m:      i32,
        n:      i32,
        A:      *mut Dtype,
        lda:    i32,
        S:      *mut Vtype,
        U:      *mut Dtype,
        ldu:    i32,
        V:      *mut Dtype,
        ldv:    i32,
        info:   *mut i32,
        params: GesVdjInfo)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "at::cuda::solver::gesvdj: not implemented for ", typeid(Dtype).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn gesvdj_batched<Dtype, Vtype>(
        handle:     CuSolverDnHandle,
        jobz:       CuSolverEigMode,
        m:          i32,
        n:          i32,
        A:          *mut Dtype,
        lda:        i32,
        S:          *mut Vtype,
        U:          *mut Dtype,
        ldu:        i32,
        V:          *mut Dtype,
        ldv:        i32,
        info:       *mut i32,
        params:     GesVdjInfo,
        batch_size: i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "at::cuda::solver::gesvdj: not implemented for ", typeid(Dtype).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf<Dtype>(
        handle: CuSolverDnHandle,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut Dtype,
        lda:    i32,
        work:   *mut Dtype,
        lwork:  i32,
        info:   *mut i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "at::cuda::solver::potrf: not implemented for ", typeid(Dtype).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_buffersize<Dtype>(
        handle: CuSolverDnHandle,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut Dtype,
        lda:    i32,
        lwork:  *mut i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "at::cuda::solver::potrf_buffersize: not implemented for ", typeid(Dtype).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_batched<Dtype>(
        handle:     CuSolverDnHandle,
        uplo:       CuBlasFillMode,
        n:          i32,
        A:          *mut *mut Dtype,
        lda:        i32,
        info:       *mut i32,
        batch_size: i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "at::cuda::solver::potrfBatched: not implemented for ", typeid(Dtype).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn geqrf_buffer_size<Scalar>(
    handle: CuSolverDnHandle,
    m:      i32,
    n:      i32,
    A:      *mut Scalar,
    lda:    i32,
    lwork:  *mut i32)  {

    todo!();
        /*
            TORCH_CHECK(
          false,
          "at::cuda::solver::geqrf_bufferSize: not implemented for ",
          typeid(Scalar).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn geqrf<Scalar>(
    handle:   CuSolverDnHandle,
    m:        i32,
    n:        i32,
    A:        *mut Scalar,
    lda:      i32,
    tau:      *mut Scalar,
    work:     *mut Scalar,
    lwork:    i32,
    dev_info: *mut i32)  {

    todo!();
    /*
            TORCH_CHECK(
          false,
          "at::cuda::solver::geqrf: not implemented for ",
          typeid(Scalar).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrs<Dtype>(
        handle:   CuSolverDnHandle,
        uplo:     CuBlasFillMode,
        n:        i32,
        nrhs:     i32,
        A:        *const Dtype,
        lda:      i32,
        B:        *mut Dtype,
        ldb:      i32,
        dev_info: *mut i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "at::cuda::solver::potrs: not implemented for ", typeid(Dtype).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrs_batched<Dtype>(
        handle:     CuSolverDnHandle,
        uplo:       CuBlasFillMode,
        n:          i32,
        nrhs:       i32,
        aarray:     &[*mut Dtype],
        lda:        i32,
        barray:     &[*mut Dtype],
        ldb:        i32,
        info:       *mut i32,
        batch_size: i32)  {

    todo!();
        /*
            cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, Dtype *Aarray[], int lda, Dtype *Barray[], int ldb, int *info, int batchSize
      TORCH_INTERNAL_ASSERT(false, "at::cuda::solver::potrsBatched: not implemented for ", typeid(Dtype).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn orgqr_buffersize<Dtype>(
        handle: CuSolverDnHandle,
        m:      i32,
        n:      i32,
        k:      i32,
        A:      *const Dtype,
        lda:    i32,
        tau:    *const Dtype,
        lwork:  *mut i32)  {

    todo!();
        /*
            TORCH_CHECK(
          false,
          "at::cuda::solver::orgqr_buffersize: not implemented for ",
          typeid(Dtype).name());
        */
}


#[cfg(CUDART_VERSION)]
pub fn orgqr<Dtype>(
        handle:   CuSolverDnHandle,
        m:        i32,
        n:        i32,
        k:        i32,
        A:        *mut Dtype,
        lda:      i32,
        tau:      *const Dtype,
        work:     *mut Dtype,
        lwork:    i32,
        dev_info: *mut i32)  {

    todo!();
        /*
            TORCH_CHECK(
          false,
          "at::cuda::solver::orgqr: not implemented for ",
          typeid(Dtype).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn ormqr_buffer_size<Dtype>(
        handle: CuSolverDnHandle,
        side:   CuBlasSideMode,
        trans:  CuBlasOperation,
        m:      i32,
        n:      i32,
        k:      i32,
        A:      *const Dtype,
        lda:    i32,
        tau:    *const Dtype,
        C:      *const Dtype,
        ldc:    i32,
        lwork:  *mut i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "at::cuda::solver::ormqr_bufferSize: not implemented for ",
          typeid(Dtype).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn ormqr<Dtype>(
        handle:   CuSolverDnHandle,
        side:     CuBlasSideMode,
        trans:    CuBlasOperation,
        m:        i32,
        n:        i32,
        k:        i32,
        A:        *const Dtype,
        lda:      i32,
        tau:      *const Dtype,
        C:        *mut Dtype,
        ldc:      i32,
        work:     *mut Dtype,
        lwork:    i32,
        dev_info: *mut i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "at::cuda::solver::ormqr: not implemented for ",
          typeid(Dtype).name());
        */
}


#[cfg(CUDART_VERSION)]
#[cfg(USE_CUSOLVER_64_BIT)]
pub fn get_cusolver_datatype<Dtype>() -> CudaDataType {

    todo!();
        /*
            TORCH_CHECK(false, "cusolver doesn't support data type ", typeid(Dtype).name());
        */
}

#[cfg(CUDART_VERSION)]
#[cfg(USE_CUSOLVER_64_BIT)]
pub fn xpotrf_buffersize(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        data_typea:                   CudaDataType,
        A:                            *const void,
        lda:                          i64,
        compute_type:                 CudaDataType,
        workspace_in_bytes_on_device: *mut usize,
        workspace_in_bytes_on_host:   *mut usize)  {
    
    todo!();
        /*
        
        */
}

#[cfg(CUDART_VERSION)]
#[cfg(USE_CUSOLVER_64_BIT)]
pub fn xpotrf(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        data_typea:                   CudaDataType,
        A:                            *mut void,
        lda:                          i64,
        compute_type:                 CudaDataType,
        buffer_on_device:             *mut void,
        workspace_in_bytes_on_device: usize,
        buffer_on_host:               *mut void,
        workspace_in_bytes_on_host:   usize,
        info:                         *mut i32)  {
    
    todo!();
        /*
        
        */
}

#[cfg(CUDART_VERSION)]
#[cfg(USE_CUSOLVER_64_BIT)]
pub fn xpotrs(
        handle:     CuSolverDnHandle,
        params:     CuSolverDnParams,
        uplo:       CuBlasFillMode,
        n:          i64,
        nrhs:       i64,
        data_typea: CudaDataType,
        A:          *const void,
        lda:        i64,
        data_typeb: CudaDataType,
        B:          *mut void,
        ldb:        i64,
        info:       *mut i32)  {
    
    todo!();
        /*
        
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevd_buffer_size<Scalar, Value = Scalar>(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *const Scalar,
        lda:    i32,
        W:      *const Value,
        lwork:  *mut i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "at::cuda::solver::syevd_bufferSize: not implemented for ",
          typeid(Scalar).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevd<Scalar, Value = Scalar>(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut Scalar,
        lda:    i32,
        W:      *mut Value,
        work:   *mut Scalar,
        lwork:  i32,
        info:   *mut i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "at::cuda::solver::syevd: not implemented for ",
          typeid(Scalar).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_buffer_size<Scalar, Value = Scalar>(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *const Scalar,
        lda:    i32,
        W:      *const Value,
        lwork:  *mut i32,
        params: SyEvjInfo)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "at::cuda::solver::syevj_bufferSize: not implemented for ",
          typeid(Scalar).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj<Scalar, Value = Scalar>(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut Scalar,
        lda:    i32,
        W:      *mut Value,
        work:   *mut Scalar,
        lwork:  i32,
        info:   *mut i32,
        params: SyEvjInfo)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "at::cuda::solver::syevj: not implemented for ",
          typeid(Scalar).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_batched_buffer_size<Scalar, Value = Scalar>(
        handle:    CuSolverDnHandle,
        jobz:      CuSolverEigMode,
        uplo:      CuBlasFillMode,
        n:         i32,
        A:         *const Scalar,
        lda:       i32,
        W:         *const Value,
        lwork:     *mut i32,
        params:    SyEvjInfo,
        batchsize: i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "at::cuda::solver::syevjBatched_bufferSize: not implemented for ",
          typeid(Scalar).name());
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_batched<Scalar, Value = Scalar>(
        handle:    CuSolverDnHandle,
        jobz:      CuSolverEigMode,
        uplo:      CuBlasFillMode,
        n:         i32,
        A:         *mut Scalar,
        lda:       i32,
        W:         *mut Value,
        work:      *mut Scalar,
        lwork:     i32,
        info:      *mut i32,
        params:    SyEvjInfo,
        batchsize: i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "at::cuda::solver::syevjBatched: not implemented for ",
          typeid(Scalar).name());
        */
}

#[cfg(CUDART_VERSION)]
#[cfg(USE_CUSOLVER_64_BIT)]
pub fn xgeqrf_buffer_size<Scalar>(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        m:                            i64,
        n:                            i64,
        A:                            *const Scalar,
        lda:                          i64,
        tau:                          *const Scalar,
        workspace_in_bytes_on_device: *mut usize,
        workspace_in_bytes_on_host:   *mut usize)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "at::cuda::solver::xgeqrf_bufferSize: not implemented for ",
          typeid(Scalar).name());
        */
}

#[cfg(CUDART_VERSION)]
#[cfg(USE_CUSOLVER_64_BIT)]
pub fn xgeqrf<Scalar>(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        m:                            i64,
        n:                            i64,
        A:                            *mut Scalar,
        lda:                          i64,
        tau:                          *mut Scalar,
        buffer_on_device:             *mut Scalar,
        workspace_in_bytes_on_device: usize,
        buffer_on_host:               *mut Scalar,
        workspace_in_bytes_on_host:   usize,
        info:                         *mut i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "at::cuda::solver::xgeqrf: not implemented for ",
          typeid(Scalar).name());
        */
}

#[cfg(CUDART_VERSION)]
#[cfg(USE_CUSOLVER_64_BIT)]
pub fn xsyevd_buffer_size<Scalar, Value = Scalar>(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        jobz:                         CuSolverEigMode,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        A:                            *const Scalar,
        lda:                          i64,
        W:                            *const Value,
        workspace_in_bytes_on_device: *mut usize,
        workspace_in_bytes_on_host:   *mut usize)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "at::cuda::solver::xsyevd_bufferSize: not implemented for ",
          typeid(Scalar).name());
        */
}

#[cfg(CUDART_VERSION)]
#[cfg(USE_CUSOLVER_64_BIT)]
pub fn xsyevd<Scalar, Value = Scalar>(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        jobz:                         CuSolverEigMode,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        A:                            *mut Scalar,
        lda:                          i64,
        W:                            *mut Value,
        buffer_on_device:             *mut Scalar,
        workspace_in_bytes_on_device: usize,
        buffer_on_host:               *mut Scalar,
        workspace_in_bytes_on_host:   usize,
        info:                         *mut i32)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          false,
          "at::cuda::solver::xsyevd: not implemented for ",
          typeid(Scalar).name());
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/CUDASolver.cpp]

#[cfg(CUDART_VERSION)]
pub fn cusolver_get_error_message(status: CuSolverStatus) -> *const u8 {
    
    todo!();
        /*
            switch (status) {
        case CUSOLVER_STATUS_SUCCESS:                     return "CUSOLVER_STATUS_SUCCES";
        case CUSOLVER_STATUS_NOT_INITIALIZED:             return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED:                return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE:               return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH:               return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED:            return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR:              return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:   return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        default:                                          return "Unknown cusolver error number";
      }
        */
}

#[cfg(CUDART_VERSION)]
pub fn getrf_double(
        handle: CuSolverDnHandle,
        m:      i32,
        n:      i32,
        da:     *mut f64,
        ldda:   i32,
        ipiv:   *mut i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            int lwork;
      TORCH_CUSOLVER_CHECK(
          cusolverDnDgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(sizeof(double)*lwork);
      TORCH_CUSOLVER_CHECK(cusolverDnDgetrf(
          handle, m, n, dA, ldda, static_cast<double*>(dataPtr.get()), ipiv, info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn getrf_float(
        handle: CuSolverDnHandle,
        m:      i32,
        n:      i32,
        da:     *mut f32,
        ldda:   i32,
        ipiv:   *mut i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            int lwork;
      TORCH_CUSOLVER_CHECK(
          cusolverDnSgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(sizeof(float)*lwork);
      TORCH_CUSOLVER_CHECK(cusolverDnSgetrf(
          handle, m, n, dA, ldda, static_cast<float*>(dataPtr.get()), ipiv, info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn getrf_complex_double(
        handle: CuSolverDnHandle,
        m:      i32,
        n:      i32,
        da:     *mut Complex<f64>,
        ldda:   i32,
        ipiv:   *mut i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            int lwork;
      TORCH_CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(
          handle, m, n, reinterpret_cast<cuDoubleComplex*>(dA), ldda, &lwork));
      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(sizeof(cuDoubleComplex) * lwork);
      TORCH_CUSOLVER_CHECK(cusolverDnZgetrf(
          handle,
          m,
          n,
          reinterpret_cast<cuDoubleComplex*>(dA),
          ldda,
          static_cast<cuDoubleComplex*>(dataPtr.get()),
          ipiv,
          info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn getrf_complex_float(
        handle: CuSolverDnHandle,
        m:      i32,
        n:      i32,
        da:     *mut Complex<f32>,
        ldda:   i32,
        ipiv:   *mut i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            int lwork;
      TORCH_CUSOLVER_CHECK(cusolverDnCgetrf_bufferSize(
          handle, m, n, reinterpret_cast<cuComplex*>(dA), ldda, &lwork));
      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(sizeof(cuComplex) * lwork);
      TORCH_CUSOLVER_CHECK(cusolverDnCgetrf(
          handle,
          m,
          n,
          reinterpret_cast<cuComplex*>(dA),
          ldda,
          static_cast<cuComplex*>(dataPtr.get()),
          ipiv,
          info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn getrs_double(
        handle: CuSolverDnHandle,
        n:      i32,
        nrhs:   i32,
        da:     *mut f64,
        lda:    i32,
        ipiv:   *mut i32,
        ret:    *mut f64,
        ldb:    i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnDgetrs(
        handle, CUBLAS_OP_N, n, nrhs, dA, lda, ipiv, ret, ldb, info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn getrs_float(
        handle: CuSolverDnHandle,
        n:      i32,
        nrhs:   i32,
        da:     *mut f32,
        lda:    i32,
        ipiv:   *mut i32,
        ret:    *mut f32,
        ldb:    i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnSgetrs(
        handle, CUBLAS_OP_N, n, nrhs, dA, lda, ipiv, ret, ldb, info));
        */
}


#[cfg(CUDART_VERSION)]
pub fn getrs_complex_double(
        handle: CuSolverDnHandle,
        n:      i32,
        nrhs:   i32,
        da:     *mut Complex<f64>,
        lda:    i32,
        ipiv:   *mut i32,
        ret:    *mut Complex<f64>,
        ldb:    i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZgetrs(
          handle,
          CUBLAS_OP_N,
          n,
          nrhs,
          reinterpret_cast<cuDoubleComplex*>(dA),
          lda,
          ipiv,
          reinterpret_cast<cuDoubleComplex*>(ret),
          ldb,
          info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn getrs_complex_float(
        handle: CuSolverDnHandle,
        n:      i32,
        nrhs:   i32,
        da:     *mut Complex<f32>,
        lda:    i32,
        ipiv:   *mut i32,
        ret:    *mut Complex<f32>,
        ldb:    i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCgetrs(
          handle,
          CUBLAS_OP_N,
          n,
          nrhs,
          reinterpret_cast<cuComplex*>(dA),
          lda,
          ipiv,
          reinterpret_cast<cuComplex*>(ret),
          ldb,
          info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn gesvdj_float(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        econ:   i32,
        m:      i32,
        n:      i32,
        A:      *mut f32,
        lda:    i32,
        S:      *mut f32,
        U:      *mut f32,
        ldu:    i32,
        V:      *mut f32,
        ldv:    i32,
        info:   *mut i32,
        params: GesVdjInfo)  {
    
    todo!();
        /*
            int lwork;
      TORCH_CUSOLVER_CHECK(cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params));

      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(sizeof(float)*lwork);

      TORCH_CUSOLVER_CHECK(cusolverDnSgesvdj(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
        static_cast<float*>(dataPtr.get()),
        lwork, info, params));
        */
}

#[cfg(CUDART_VERSION)]
pub fn gesvdj_double(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        econ:   i32,
        m:      i32,
        n:      i32,
        A:      *mut f64,
        lda:    i32,
        S:      *mut f64,
        U:      *mut f64,
        ldu:    i32,
        V:      *mut f64,
        ldv:    i32,
        info:   *mut i32,
        params: GesVdjInfo)  {
    
    todo!();
        /*
            int lwork;
      TORCH_CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params));

      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(sizeof(double)*lwork);

      TORCH_CUSOLVER_CHECK(cusolverDnDgesvdj(
        handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
        static_cast<double*>(dataPtr.get()),
        lwork, info, params));
        */
}

#[cfg(CUDART_VERSION)]
pub fn gesvdj_complex_float(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        econ:   i32,
        m:      i32,
        n:      i32,
        A:      *mut Complex<f32>,
        lda:    i32,
        S:      *mut f32,
        U:      *mut Complex<f32>,
        ldu:    i32,
        V:      *mut Complex<f32>,
        ldv:    i32,
        info:   *mut i32,
        params: GesVdjInfo)  {
    
    todo!();
        /*
            int lwork;
      TORCH_CUSOLVER_CHECK(cusolverDnCgesvdj_bufferSize(
        handle, jobz, econ, m, n,
        reinterpret_cast<cuComplex*>(A),
        lda, S,
        reinterpret_cast<cuComplex*>(U),
        ldu,
        reinterpret_cast<cuComplex*>(V),
        ldv, &lwork, params));

      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(sizeof(cuComplex)*lwork);

      TORCH_CUSOLVER_CHECK(cusolverDnCgesvdj(
        handle, jobz, econ, m, n,
        reinterpret_cast<cuComplex*>(A),
        lda, S,
        reinterpret_cast<cuComplex*>(U),
        ldu,
        reinterpret_cast<cuComplex*>(V),
        ldv,
        static_cast<cuComplex*>(dataPtr.get()),
        lwork, info, params));
        */
}

#[cfg(CUDART_VERSION)]
pub fn gesvdj_complex_double(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        econ:   i32,
        m:      i32,
        n:      i32,
        A:      *mut Complex<f64>,
        lda:    i32,
        S:      *mut f64,
        U:      *mut Complex<f64>,
        ldu:    i32,
        V:      *mut Complex<f64>,
        ldv:    i32,
        info:   *mut i32,
        params: GesVdjInfo)  {
    
    todo!();
        /*
            int lwork;
      TORCH_CUSOLVER_CHECK(cusolverDnZgesvdj_bufferSize(
        handle, jobz, econ, m, n,
        reinterpret_cast<cuDoubleComplex*>(A),
        lda, S,
        reinterpret_cast<cuDoubleComplex*>(U),
        ldu,
        reinterpret_cast<cuDoubleComplex*>(V),
        ldv, &lwork, params));

      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(sizeof(cuDoubleComplex)*lwork);

      TORCH_CUSOLVER_CHECK(cusolverDnZgesvdj(
        handle, jobz, econ, m, n,
        reinterpret_cast<cuDoubleComplex*>(A),
        lda, S,
        reinterpret_cast<cuDoubleComplex*>(U),
        ldu,
        reinterpret_cast<cuDoubleComplex*>(V),
        ldv,
        static_cast<cuDoubleComplex*>(dataPtr.get()),
        lwork, info, params));
        */
}

#[cfg(CUDART_VERSION)]
pub fn gesvdj_batched_float(
        handle:     CuSolverDnHandle,
        jobz:       CuSolverEigMode,
        m:          i32,
        n:          i32,
        A:          *mut f32,
        lda:        i32,
        S:          *mut f32,
        U:          *mut f32,
        ldu:        i32,
        V:          *mut f32,
        ldv:        i32,
        info:       *mut i32,
        params:     GesVdjInfo,
        batch_size: i32)  {
    
    todo!();
        /*
            int lwork;
      TORCH_CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize));

      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(sizeof(float)*lwork);

      TORCH_CUSOLVER_CHECK(cusolverDnSgesvdjBatched(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
        static_cast<float*>(dataPtr.get()),
        lwork, info, params, batchSize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn gesvdj_batched_double(
        handle:     CuSolverDnHandle,
        jobz:       CuSolverEigMode,
        m:          i32,
        n:          i32,
        A:          *mut f64,
        lda:        i32,
        S:          *mut f64,
        U:          *mut f64,
        ldu:        i32,
        V:          *mut f64,
        ldv:        i32,
        info:       *mut i32,
        params:     GesVdjInfo,
        batch_size: i32)  {
    
    todo!();
        /*
            int lwork;
      TORCH_CUSOLVER_CHECK(cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize));

      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(sizeof(double)*lwork);

      TORCH_CUSOLVER_CHECK(cusolverDnDgesvdjBatched(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
        static_cast<double*>(dataPtr.get()),
        lwork, info, params, batchSize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn gesvdj_batched_complex_float(
        handle:     CuSolverDnHandle,
        jobz:       CuSolverEigMode,
        m:          i32,
        n:          i32,
        A:          *mut Complex<f32>,
        lda:        i32,
        S:          *mut f32,
        U:          *mut Complex<f32>,
        ldu:        i32,
        V:          *mut Complex<f32>,
        ldv:        i32,
        info:       *mut i32,
        params:     GesVdjInfo,
        batch_size: i32)  {
    
    todo!();
        /*
            int lwork;
      TORCH_CUSOLVER_CHECK(cusolverDnCgesvdjBatched_bufferSize(
        handle, jobz, m, n,
        reinterpret_cast<cuComplex*>(A),
        lda, S,
        reinterpret_cast<cuComplex*>(U),
        ldu,
        reinterpret_cast<cuComplex*>(V),
        ldv, &lwork, params, batchSize));

      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(sizeof(cuComplex)*lwork);

      TORCH_CUSOLVER_CHECK(cusolverDnCgesvdjBatched(
        handle, jobz, m, n,
        reinterpret_cast<cuComplex*>(A),
        lda, S,
        reinterpret_cast<cuComplex*>(U),
        ldu,
        reinterpret_cast<cuComplex*>(V),
        ldv,
        static_cast<cuComplex*>(dataPtr.get()),
        lwork, info, params, batchSize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn gesvdj_batched_complex_double(
        handle:     CuSolverDnHandle,
        jobz:       CuSolverEigMode,
        m:          i32,
        n:          i32,
        A:          *mut Complex<f64>,
        lda:        i32,
        S:          *mut f64,
        U:          *mut Complex<f64>,
        ldu:        i32,
        V:          *mut Complex<f64>,
        ldv:        i32,
        info:       *mut i32,
        params:     GesVdjInfo,
        batch_size: i32)  {
    
    todo!();
        /*
            int lwork;
      TORCH_CUSOLVER_CHECK(cusolverDnZgesvdjBatched_bufferSize(
        handle, jobz, m, n,
        reinterpret_cast<cuDoubleComplex*>(A),
        lda, S,
        reinterpret_cast<cuDoubleComplex*>(U),
        ldu,
        reinterpret_cast<cuDoubleComplex*>(V),
        ldv, &lwork, params, batchSize));

      auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
      auto dataPtr = allocator.allocate(sizeof(cuDoubleComplex)*lwork);

      TORCH_CUSOLVER_CHECK(cusolverDnZgesvdjBatched(
        handle, jobz, m, n,
        reinterpret_cast<cuDoubleComplex*>(A),
        lda, S,
        reinterpret_cast<cuDoubleComplex*>(U),
        ldu,
        reinterpret_cast<cuDoubleComplex*>(V),
        ldv,
        static_cast<cuDoubleComplex*>(dataPtr.get()),
        lwork, info, params, batchSize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_float(
        handle: CuSolverDnHandle,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut f32,
        lda:    i32,
        work:   *mut f32,
        lwork:  i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnSpotrf(
        handle, uplo, n, A, lda, work, lwork, info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_double(
        handle: CuSolverDnHandle,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut f64,
        lda:    i32,
        work:   *mut f64,
        lwork:  i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnDpotrf(
        handle, uplo, n, A, lda, work, lwork, info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_complex_float(
        handle: CuSolverDnHandle,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut Complex<f32>,
        lda:    i32,
        work:   *mut Complex<f32>,
        lwork:  i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCpotrf(
        handle,
        uplo,
        n,
        reinterpret_cast<cuComplex*>(A),
        lda,
        reinterpret_cast<cuComplex*>(work),
        lwork,
        info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_complex_double(
        handle: CuSolverDnHandle,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut Complex<f64>,
        lda:    i32,
        work:   *mut Complex<f64>,
        lwork:  i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZpotrf(
        handle,
        uplo,
        n,
        reinterpret_cast<cuDoubleComplex*>(A),
        lda,
        reinterpret_cast<cuDoubleComplex*>(work),
        lwork,
        info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_buffersize_float(
        handle: CuSolverDnHandle,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut f32,
        lda:    i32,
        lwork:  *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_buffersize_double(
        handle: CuSolverDnHandle,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut f64,
        lda:    i32,
        lwork:  *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_buffersize_complex_float(
        handle: CuSolverDnHandle,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut Complex<f32>,
        lda:    i32,
        lwork:  *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCpotrf_bufferSize(
        handle, uplo, n,
        reinterpret_cast<cuComplex*>(A),
        lda, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_buffersize_complex_double(
        handle: CuSolverDnHandle,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut Complex<f64>,
        lda:    i32,
        lwork:  *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZpotrf_bufferSize(
        handle, uplo, n,
        reinterpret_cast<cuDoubleComplex*>(A),
        lda, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_batched_float(
        handle:     CuSolverDnHandle,
        uplo:       CuBlasFillMode,
        n:          i32,
        A:          *mut *mut f32,
        lda:        i32,
        info:       *mut i32,
        batch_size: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnSpotrfBatched(handle, uplo, n, A, lda, info, batchSize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_batched_double(
        handle:     CuSolverDnHandle,
        uplo:       CuBlasFillMode,
        n:          i32,
        A:          *mut *mut f64,
        lda:        i32,
        info:       *mut i32,
        batch_size: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnDpotrfBatched(handle, uplo, n, A, lda, info, batchSize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_batched_complex_float(
        handle:     CuSolverDnHandle,
        uplo:       CuBlasFillMode,
        n:          i32,
        A:          *mut *mut Complex<f32>,
        lda:        i32,
        info:       *mut i32,
        batch_size: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCpotrfBatched(
        handle, uplo, n,
        reinterpret_cast<cuComplex**>(A),
        lda, info, batchSize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrf_batched_complex_double(
        handle:     CuSolverDnHandle,
        uplo:       CuBlasFillMode,
        n:          i32,
        A:          *mut *mut Complex<f64>,
        lda:        i32,
        info:       *mut i32,
        batch_size: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZpotrfBatched(
        handle, uplo, n,
        reinterpret_cast<cuDoubleComplex**>(A),
        lda, info, batchSize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn geqrf_buffer_size_float()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn geqrf_buffer_size_double()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn geqrf_buffer_size_complex_float()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCgeqrf_bufferSize(
          handle, m, n, reinterpret_cast<cuComplex*>(A), lda, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn geqrf_buffer_size_complex_double()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZgeqrf_bufferSize(
          handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn geqrf_float()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnSgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn geqrf_double()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnDgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn geqrf_complex_float()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCgeqrf(
          handle,
          m,
          n,
          reinterpret_cast<cuComplex*>(A),
          lda,
          reinterpret_cast<cuComplex*>(tau),
          reinterpret_cast<cuComplex*>(work),
          lwork,
          devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn geqrf_complex_double()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZgeqrf(
          handle,
          m,
          n,
          reinterpret_cast<cuDoubleComplex*>(A),
          lda,
          reinterpret_cast<cuDoubleComplex*>(tau),
          reinterpret_cast<cuDoubleComplex*>(work),
          lwork,
          devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrs_float(
        handle:   CuSolverDnHandle,
        uplo:     CuBlasFillMode,
        n:        i32,
        nrhs:     i32,
        A:        *const f32,
        lda:      i32,
        B:        *mut f32,
        ldb:      i32,
        dev_info: *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrs_double(
        handle:   CuSolverDnHandle,
        uplo:     CuBlasFillMode,
        n:        i32,
        nrhs:     i32,
        A:        *const f64,
        lda:      i32,
        B:        *mut f64,
        ldb:      i32,
        dev_info: *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrs_complex_float(
        handle:   CuSolverDnHandle,
        uplo:     CuBlasFillMode,
        n:        i32,
        nrhs:     i32,
        A:        *const Complex<f32>,
        lda:      i32,
        B:        *mut Complex<f32>,
        ldb:      i32,
        dev_info: *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCpotrs(
        handle, uplo, n, nrhs,
        reinterpret_cast<const cuComplex*>(A),
        lda,
        reinterpret_cast<cuComplex*>(B),
        ldb, devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrs_complex_double(
        handle:   CuSolverDnHandle,
        uplo:     CuBlasFillMode,
        n:        i32,
        nrhs:     i32,
        A:        *const Complex<f64>,
        lda:      i32,
        B:        *mut Complex<f64>,
        ldb:      i32,
        dev_info: *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZpotrs(
        handle, uplo, n, nrhs,
        reinterpret_cast<const cuDoubleComplex*>(A),
        lda,
        reinterpret_cast<cuDoubleComplex*>(B),
        ldb, devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrs_batched_float(
        handle:     CuSolverDnHandle,
        uplo:       CuBlasFillMode,
        n:          i32,
        nrhs:       i32,
        aarray:     &[*mut f32],
        lda:        i32,
        barray:     &[*mut f32],
        ldb:        i32,
        info:       *mut i32,
        batch_size: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnSpotrsBatched(handle, uplo, n, nrhs, Aarray, lda, Barray, ldb, info, batchSize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrs_batched_double(
        handle:     CuSolverDnHandle,
        uplo:       CuBlasFillMode,
        n:          i32,
        nrhs:       i32,
        aarray:     &[*mut f64],
        lda:        i32,
        barray:     &[*mut f64],
        ldb:        i32,
        info:       *mut i32,
        batch_size: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnDpotrsBatched(handle, uplo, n, nrhs, Aarray, lda, Barray, ldb, info, batchSize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrs_batched_complex_float(
        handle:     CuSolverDnHandle,
        uplo:       CuBlasFillMode,
        n:          i32,
        nrhs:       i32,
        aarray:     &[*mut Complex<f32>],
        lda:        i32,
        barray:     &[*mut Complex<f32>],
        ldb:        i32,
        info:       *mut i32,
        batch_size: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCpotrsBatched(
        handle, uplo, n, nrhs,
        reinterpret_cast<cuComplex**>(Aarray),
        lda,
        reinterpret_cast<cuComplex**>(Barray),
        ldb, info, batchSize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn potrs_batched_complex_double(
        handle:     CuSolverDnHandle,
        uplo:       CuBlasFillMode,
        n:          i32,
        nrhs:       i32,
        aarray:     &[*mut Complex<f64>],
        lda:        i32,
        barray:     &[*mut Complex<f64>],
        ldb:        i32,
        info:       *mut i32,
        batch_size: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZpotrsBatched(
        handle, uplo, n, nrhs,
        reinterpret_cast<cuDoubleComplex**>(Aarray),
        lda,
        reinterpret_cast<cuDoubleComplex**>(Barray),
        ldb, info, batchSize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn orgqr_buffersize_float(
        handle: CuSolverDnHandle,
        m:      i32,
        n:      i32,
        k:      i32,
        A:      *const f32,
        lda:    i32,
        tau:    *const f32,
        lwork:  *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn orgqr_buffersize_double(
        handle: CuSolverDnHandle,
        m:      i32,
        n:      i32,
        k:      i32,
        A:      *const f64,
        lda:    i32,
        tau:    *const f64,
        lwork:  *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn orgqr_buffersize_complex_float(
        handle: CuSolverDnHandle,
        m:      i32,
        n:      i32,
        k:      i32,
        A:      *const Complex<f32>,
        lda:    i32,
        tau:    *const Complex<f32>,
        lwork:  *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCungqr_bufferSize(
          handle,
          m, n, k,
          reinterpret_cast<const cuComplex*>(A), lda,
          reinterpret_cast<const cuComplex*>(tau), lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn orgqr_buffersize_complex_double(
        handle: CuSolverDnHandle,
        m:      i32,
        n:      i32,
        k:      i32,
        A:      *const Complex<f64>,
        lda:    i32,
        tau:    *const Complex<f64>,
        lwork:  *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZungqr_bufferSize(
          handle,
          m, n, k,
          reinterpret_cast<const cuDoubleComplex*>(A), lda,
          reinterpret_cast<const cuDoubleComplex*>(tau), lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn orgqr_float(
        handle:   CuSolverDnHandle,
        m:        i32,
        n:        i32,
        k:        i32,
        A:        *mut f32,
        lda:      i32,
        tau:      *const f32,
        work:     *mut f32,
        lwork:    i32,
        dev_info: *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn orgqr_double(
        handle:   CuSolverDnHandle,
        m:        i32,
        n:        i32,
        k:        i32,
        A:        *mut f64,
        lda:      i32,
        tau:      *const f64,
        work:     *mut f64,
        lwork:    i32,
        dev_info: *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn orgqr_complex_float(
        handle:   CuSolverDnHandle,
        m:        i32,
        n:        i32,
        k:        i32,
        A:        *mut Complex<f32>,
        lda:      i32,
        tau:      *const Complex<f32>,
        work:     *mut Complex<f32>,
        lwork:    i32,
        dev_info: *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCungqr(
          handle,
          m, n, k,
          reinterpret_cast<cuComplex*>(A), lda,
          reinterpret_cast<const cuComplex*>(tau),
          reinterpret_cast<cuComplex*>(work), lwork,
          devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn orgqr_complex_double(
        handle:   CuSolverDnHandle,
        m:        i32,
        n:        i32,
        k:        i32,
        A:        *mut Complex<f64>,
        lda:      i32,
        tau:      *const Complex<f64>,
        work:     *mut Complex<f64>,
        lwork:    i32,
        dev_info: *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZungqr(
          handle,
          m, n, k,
          reinterpret_cast<cuDoubleComplex*>(A), lda,
          reinterpret_cast<const cuDoubleComplex*>(tau),
          reinterpret_cast<cuDoubleComplex*>(work), lwork,
          devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn ormqr_buffer_size_float()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn ormqr_buffer_size_double()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn ormqr_buffer_size_complex_float()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCunmqr_bufferSize(
          handle, side, trans,
          m, n, k,
          reinterpret_cast<const cuComplex*>(A), lda,
          reinterpret_cast<const cuComplex*>(tau),
          reinterpret_cast<const cuComplex*>(C), ldc,
          lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn ormqr_buffer_size_complex_double()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZunmqr_bufferSize(
          handle, side, trans,
          m, n, k,
          reinterpret_cast<const cuDoubleComplex*>(A), lda,
          reinterpret_cast<const cuDoubleComplex*>(tau),
          reinterpret_cast<const cuDoubleComplex*>(C), ldc,
          lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn ormqr_float()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn ormqr_double()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn ormqr_complex_float()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCunmqr(
          handle, side, trans,
          m, n, k,
          reinterpret_cast<const cuComplex*>(A), lda,
          reinterpret_cast<const cuComplex*>(tau),
          reinterpret_cast<cuComplex*>(C), ldc,
          reinterpret_cast<cuComplex*>(work), lwork,
          devInfo));
        */
}

#[cfg(CUDART_VERSION)]
pub fn ormqr_complex_double()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZunmqr(
          handle, side, trans,
          m, n, k,
          reinterpret_cast<const cuDoubleComplex*>(A), lda,
          reinterpret_cast<const cuDoubleComplex*>(tau),
          reinterpret_cast<cuDoubleComplex*>(C), ldc,
          reinterpret_cast<cuDoubleComplex*>(work), lwork,
          devInfo));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn get_cusolver_datatype_float() -> CudaDataType {
    
    todo!();
        /*
            return CUDA_R_32F;
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn get_cusolver_datatype_double() -> CudaDataType {
    
    todo!();
        /*
            return CUDA_R_64F;
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn get_cusolver_datatype_complex_float() -> CudaDataType {
    
    todo!();
        /*
            return CUDA_C_32F;
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn get_cusolver_datatype_complex_double() -> CudaDataType {
    
    todo!();
        /*
            return CUDA_C_64F;
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xpotrf_buffersize(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        data_typea:                   CudaDataType,
        A:                            *const void,
        lda:                          i64,
        compute_type:                 CudaDataType,
        workspace_in_bytes_on_device: *mut usize,
        workspace_in_bytes_on_host:   *mut usize)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
        handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost
      ));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xpotrf(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        data_typea:                   CudaDataType,
        A:                            *mut void,
        lda:                          i64,
        compute_type:                 CudaDataType,
        buffer_on_device:             *mut void,
        workspace_in_bytes_on_device: usize,
        buffer_on_host:               *mut void,
        workspace_in_bytes_on_host:   usize,
        info:                         *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXpotrf(
        handle, params, uplo, n, dataTypeA, A, lda, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info
      ));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevd_buffer_size_float(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *const f32,
        lda:    i32,
        W:      *const f32,
        lwork:  *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevd_buffer_size_double(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *const f64,
        lda:    i32,
        W:      *const f64,
        lwork:  *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevd_buffer_size_complex_float2(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *const Complex<f32>,
        lda:    i32,
        W:      *const f32,
        lwork:  *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCheevd_bufferSize(
          handle,
          jobz,
          uplo,
          n,
          reinterpret_cast<const cuComplex*>(A),
          lda,
          W,
          lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevd_buffer_size_complex_double2(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *const Complex<f64>,
        lda:    i32,
        W:      *const f64,
        lwork:  *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(
          handle,
          jobz,
          uplo,
          n,
          reinterpret_cast<const cuDoubleComplex*>(A),
          lda,
          W,
          lwork));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevd_float(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut f32,
        lda:    i32,
        W:      *mut f32,
        work:   *mut f32,
        lwork:  i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevd_double(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut f64,
        lda:    i32,
        W:      *mut f64,
        work:   *mut f64,
        lwork:  i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(
          cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevd_complex_float2(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut Complex<f32>,
        lda:    i32,
        W:      *mut f32,
        work:   *mut Complex<f32>,
        lwork:  i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCheevd(
          handle,
          jobz,
          uplo,
          n,
          reinterpret_cast<cuComplex*>(A),
          lda,
          W,
          reinterpret_cast<cuComplex*>(work),
          lwork,
          info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevd_complex_double2(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut Complex<f64>,
        lda:    i32,
        W:      *mut f64,
        work:   *mut Complex<f64>,
        lwork:  i32,
        info:   *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZheevd(
          handle,
          jobz,
          uplo,
          n,
          reinterpret_cast<cuDoubleComplex*>(A),
          lda,
          W,
          reinterpret_cast<cuDoubleComplex*>(work),
          lwork,
          info));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_buffer_size_float(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *const f32,
        lda:    i32,
        W:      *const f32,
        lwork:  *mut i32,
        params: SyEvjInfo)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnSsyevj_bufferSize(
          handle, jobz, uplo, n, A, lda, W, lwork, params));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_buffer_size_double(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *const f64,
        lda:    i32,
        W:      *const f64,
        lwork:  *mut i32,
        params: SyEvjInfo)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnDsyevj_bufferSize(
          handle, jobz, uplo, n, A, lda, W, lwork, params));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_buffer_size_complex_float2(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *const Complex<f32>,
        lda:    i32,
        W:      *const f32,
        lwork:  *mut i32,
        params: SyEvjInfo)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCheevj_bufferSize(
          handle,
          jobz,
          uplo,
          n,
          reinterpret_cast<const cuComplex*>(A),
          lda,
          W,
          lwork,
          params));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_buffer_size_complex_double2(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *const Complex<f64>,
        lda:    i32,
        W:      *const f64,
        lwork:  *mut i32,
        params: SyEvjInfo)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZheevj_bufferSize(
          handle,
          jobz,
          uplo,
          n,
          reinterpret_cast<const cuDoubleComplex*>(A),
          lda,
          W,
          lwork,
          params));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_float(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut f32,
        lda:    i32,
        W:      *mut f32,
        work:   *mut f32,
        lwork:  i32,
        info:   *mut i32,
        params: SyEvjInfo)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnSsyevj(
          handle, jobz, uplo, n, A, lda, W, work, lwork, info, params));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_double(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut f64,
        lda:    i32,
        W:      *mut f64,
        work:   *mut f64,
        lwork:  i32,
        info:   *mut i32,
        params: SyEvjInfo)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnDsyevj(
          handle, jobz, uplo, n, A, lda, W, work, lwork, info, params));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_complex_float2(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut Complex<f32>,
        lda:    i32,
        W:      *mut f32,
        work:   *mut Complex<f32>,
        lwork:  i32,
        info:   *mut i32,
        params: SyEvjInfo)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCheevj(
          handle,
          jobz,
          uplo,
          n,
          reinterpret_cast<cuComplex*>(A),
          lda,
          W,
          reinterpret_cast<cuComplex*>(work),
          lwork,
          info,
          params));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_complex_double2(
        handle: CuSolverDnHandle,
        jobz:   CuSolverEigMode,
        uplo:   CuBlasFillMode,
        n:      i32,
        A:      *mut Complex<f64>,
        lda:    i32,
        W:      *mut f64,
        work:   *mut Complex<f64>,
        lwork:  i32,
        info:   *mut i32,
        params: SyEvjInfo)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZheevj(
          handle,
          jobz,
          uplo,
          n,
          reinterpret_cast<cuDoubleComplex*>(A),
          lda,
          W,
          reinterpret_cast<cuDoubleComplex*>(work),
          lwork,
          info,
          params));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_batched_buffer_size_float(
        handle:    CuSolverDnHandle,
        jobz:      CuSolverEigMode,
        uplo:      CuBlasFillMode,
        n:         i32,
        A:         *const f32,
        lda:       i32,
        W:         *const f32,
        lwork:     *mut i32,
        params:    SyEvjInfo,
        batchsize: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnSsyevjBatched_bufferSize(
          handle, jobz, uplo, n, A, lda, W, lwork, params, batchsize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_batched_buffer_size_double(
        handle:    CuSolverDnHandle,
        jobz:      CuSolverEigMode,
        uplo:      CuBlasFillMode,
        n:         i32,
        A:         *const f64,
        lda:       i32,
        W:         *const f64,
        lwork:     *mut i32,
        params:    SyEvjInfo,
        batchsize: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnDsyevjBatched_bufferSize(
          handle, jobz, uplo, n, A, lda, W, lwork, params, batchsize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_batched_buffer_size_complex_float2(
        handle:    CuSolverDnHandle,
        jobz:      CuSolverEigMode,
        uplo:      CuBlasFillMode,
        n:         i32,
        A:         *const Complex<f32>,
        lda:       i32,
        W:         *const f32,
        lwork:     *mut i32,
        params:    SyEvjInfo,
        batchsize: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCheevjBatched_bufferSize(
          handle,
          jobz,
          uplo,
          n,
          reinterpret_cast<const cuComplex*>(A),
          lda,
          W,
          lwork,
          params,
          batchsize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_batched_buffer_size_complex_double2(
        handle:    CuSolverDnHandle,
        jobz:      CuSolverEigMode,
        uplo:      CuBlasFillMode,
        n:         i32,
        A:         *const Complex<f64>,
        lda:       i32,
        W:         *const f64,
        lwork:     *mut i32,
        params:    SyEvjInfo,
        batchsize: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(
          handle,
          jobz,
          uplo,
          n,
          reinterpret_cast<const cuDoubleComplex*>(A),
          lda,
          W,
          lwork,
          params,
          batchsize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_batched_float(
        handle:    CuSolverDnHandle,
        jobz:      CuSolverEigMode,
        uplo:      CuBlasFillMode,
        n:         i32,
        A:         *mut f32,
        lda:       i32,
        W:         *mut f32,
        work:      *mut f32,
        lwork:     i32,
        info:      *mut i32,
        params:    SyEvjInfo,
        batchsize: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnSsyevjBatched(
          handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchsize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_batched_double(
        handle:    CuSolverDnHandle,
        jobz:      CuSolverEigMode,
        uplo:      CuBlasFillMode,
        n:         i32,
        A:         *mut f64,
        lda:       i32,
        W:         *mut f64,
        work:      *mut f64,
        lwork:     i32,
        info:      *mut i32,
        params:    SyEvjInfo,
        batchsize: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnDsyevjBatched(
          handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchsize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_batched_complex_float2(
        handle:    CuSolverDnHandle,
        jobz:      CuSolverEigMode,
        uplo:      CuBlasFillMode,
        n:         i32,
        A:         *mut Complex<f32>,
        lda:       i32,
        W:         *mut f32,
        work:      *mut Complex<f32>,
        lwork:     i32,
        info:      *mut i32,
        params:    SyEvjInfo,
        batchsize: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCheevjBatched(
          handle,
          jobz,
          uplo,
          n,
          reinterpret_cast<cuComplex*>(A),
          lda,
          W,
          reinterpret_cast<cuComplex*>(work),
          lwork,
          info,
          params,
          batchsize));
        */
}

#[cfg(CUDART_VERSION)]
pub fn syevj_batched_complex_double2(
        handle:    CuSolverDnHandle,
        jobz:      CuSolverEigMode,
        uplo:      CuBlasFillMode,
        n:         i32,
        A:         *mut Complex<f64>,
        lda:       i32,
        W:         *mut f64,
        work:      *mut Complex<f64>,
        lwork:     i32,
        info:      *mut i32,
        params:    SyEvjInfo,
        batchsize: i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnZheevjBatched(
          handle,
          jobz,
          uplo,
          n,
          reinterpret_cast<cuDoubleComplex*>(A),
          lda,
          W,
          reinterpret_cast<cuDoubleComplex*>(work),
          lwork,
          info,
          params,
          batchsize));
        */
}


#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xpotrs(
        handle:     CuSolverDnHandle,
        params:     CuSolverDnParams,
        uplo:       CuBlasFillMode,
        n:          i64,
        nrhs:       i64,
        data_typea: CudaDataType,
        A:          *const void,
        lda:        i64,
        data_typeb: CudaDataType,
        B:          *mut void,
        ldb:        i64,
        info:       *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXpotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xgeqrf_buffer_size_float()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
          handle,
          params,
          m,
          n,
          CUDA_R_32F,
          reinterpret_cast<const void*>(A),
          lda,
          CUDA_R_32F,
          reinterpret_cast<const void*>(tau),
          CUDA_R_32F,
          workspaceInBytesOnDevice,
          workspaceInBytesOnHost));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xgeqrf_buffer_size_double()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
          handle,
          params,
          m,
          n,
          CUDA_R_64F,
          reinterpret_cast<const void*>(A),
          lda,
          CUDA_R_64F,
          reinterpret_cast<const void*>(tau),
          CUDA_R_64F,
          workspaceInBytesOnDevice,
          workspaceInBytesOnHost));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xgeqrf_buffer_size_complex_float()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
          handle,
          params,
          m,
          n,
          CUDA_C_32F,
          reinterpret_cast<const void*>(A),
          lda,
          CUDA_C_32F,
          reinterpret_cast<const void*>(tau),
          CUDA_C_32F,
          workspaceInBytesOnDevice,
          workspaceInBytesOnHost));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xgeqrf_buffer_size_complex_double()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
          handle,
          params,
          m,
          n,
          CUDA_C_64F,
          reinterpret_cast<const void*>(A),
          lda,
          CUDA_C_64F,
          reinterpret_cast<const void*>(tau),
          CUDA_C_64F,
          workspaceInBytesOnDevice,
          workspaceInBytesOnHost));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xgeqrf_float()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf(
          handle,
          params,
          m,
          n,
          CUDA_R_32F,
          reinterpret_cast<void*>(A),
          lda,
          CUDA_R_32F,
          reinterpret_cast<void*>(tau),
          CUDA_R_32F,
          reinterpret_cast<void*>(bufferOnDevice),
          workspaceInBytesOnDevice,
          reinterpret_cast<void*>(bufferOnHost),
          workspaceInBytesOnHost,
          info));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xgeqrf_double()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf(
          handle,
          params,
          m,
          n,
          CUDA_R_64F,
          reinterpret_cast<void*>(A),
          lda,
          CUDA_R_64F,
          reinterpret_cast<void*>(tau),
          CUDA_R_64F,
          reinterpret_cast<void*>(bufferOnDevice),
          workspaceInBytesOnDevice,
          reinterpret_cast<void*>(bufferOnHost),
          workspaceInBytesOnHost,
          info));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xgeqrf_complex_float()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf(
          handle,
          params,
          m,
          n,
          CUDA_C_32F,
          reinterpret_cast<void*>(A),
          lda,
          CUDA_C_32F,
          reinterpret_cast<void*>(tau),
          CUDA_C_32F,
          reinterpret_cast<void*>(bufferOnDevice),
          workspaceInBytesOnDevice,
          reinterpret_cast<void*>(bufferOnHost),
          workspaceInBytesOnHost,
          info));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xgeqrf_complex_double()  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf(
          handle,
          params,
          m,
          n,
          CUDA_C_64F,
          reinterpret_cast<void*>(A),
          lda,
          CUDA_C_64F,
          reinterpret_cast<void*>(tau),
          CUDA_C_64F,
          reinterpret_cast<void*>(bufferOnDevice),
          workspaceInBytesOnDevice,
          reinterpret_cast<void*>(bufferOnHost),
          workspaceInBytesOnHost,
          info));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xsyevd_buffer_size_float(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        jobz:                         CuSolverEigMode,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        A:                            *const f32,
        lda:                          i64,
        W:                            *const f32,
        workspace_in_bytes_on_device: *mut usize,
        workspace_in_bytes_on_host:   *mut usize)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
          handle,
          params,
          jobz,
          uplo,
          n,
          CUDA_R_32F,
          reinterpret_cast<const void*>(A),
          lda,
          CUDA_R_32F,
          reinterpret_cast<const void*>(W),
          CUDA_R_32F,
          workspaceInBytesOnDevice,
          workspaceInBytesOnHost));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xsyevd_buffer_size_double(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        jobz:                         CuSolverEigMode,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        A:                            *const f64,
        lda:                          i64,
        W:                            *const f64,
        workspace_in_bytes_on_device: *mut usize,
        workspace_in_bytes_on_host:   *mut usize)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
          handle,
          params,
          jobz,
          uplo,
          n,
          CUDA_R_64F,
          reinterpret_cast<const void*>(A),
          lda,
          CUDA_R_64F,
          reinterpret_cast<const void*>(W),
          CUDA_R_64F,
          workspaceInBytesOnDevice,
          workspaceInBytesOnHost));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xsyevd_buffer_size_complex_float2(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        jobz:                         CuSolverEigMode,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        A:                            *const Complex<f32>,
        lda:                          i64,
        W:                            *const f32,
        workspace_in_bytes_on_device: *mut usize,
        workspace_in_bytes_on_host:   *mut usize)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
          handle,
          params,
          jobz,
          uplo,
          n,
          CUDA_C_32F,
          reinterpret_cast<const void*>(A),
          lda,
          CUDA_R_32F,
          reinterpret_cast<const void*>(W),
          CUDA_C_32F,
          workspaceInBytesOnDevice,
          workspaceInBytesOnHost));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xsyevd_buffer_size_complex_double2(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        jobz:                         CuSolverEigMode,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        A:                            *const Complex<f64>,
        lda:                          i64,
        W:                            *const f64,
        workspace_in_bytes_on_device: *mut usize,
        workspace_in_bytes_on_host:   *mut usize)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
          handle,
          params,
          jobz,
          uplo,
          n,
          CUDA_C_64F,
          reinterpret_cast<const void*>(A),
          lda,
          CUDA_R_64F,
          reinterpret_cast<const void*>(W),
          CUDA_C_64F,
          workspaceInBytesOnDevice,
          workspaceInBytesOnHost));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xsyevd_float(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        jobz:                         CuSolverEigMode,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        A:                            *mut f32,
        lda:                          i64,
        W:                            *mut f32,
        buffer_on_device:             *mut f32,
        workspace_in_bytes_on_device: usize,
        buffer_on_host:               *mut f32,
        workspace_in_bytes_on_host:   usize,
        info:                         *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
          handle,
          params,
          jobz,
          uplo,
          n,
          CUDA_R_32F,
          reinterpret_cast<void*>(A),
          lda,
          CUDA_R_32F,
          reinterpret_cast<void*>(W),
          CUDA_R_32F,
          reinterpret_cast<void*>(bufferOnDevice),
          workspaceInBytesOnDevice,
          reinterpret_cast<void*>(bufferOnHost),
          workspaceInBytesOnHost,
          info));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xsyevd_double(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        jobz:                         CuSolverEigMode,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        A:                            *mut f64,
        lda:                          i64,
        W:                            *mut f64,
        buffer_on_device:             *mut f64,
        workspace_in_bytes_on_device: usize,
        buffer_on_host:               *mut f64,
        workspace_in_bytes_on_host:   usize,
        info:                         *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
          handle,
          params,
          jobz,
          uplo,
          n,
          CUDA_R_64F,
          reinterpret_cast<void*>(A),
          lda,
          CUDA_R_64F,
          reinterpret_cast<void*>(W),
          CUDA_R_64F,
          reinterpret_cast<void*>(bufferOnDevice),
          workspaceInBytesOnDevice,
          reinterpret_cast<void*>(bufferOnHost),
          workspaceInBytesOnHost,
          info));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xsyevd_complex_float2(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        jobz:                         CuSolverEigMode,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        A:                            *mut Complex<f32>,
        lda:                          i64,
        W:                            *mut f32,
        buffer_on_device:             *mut Complex<f32>,
        workspace_in_bytes_on_device: usize,
        buffer_on_host:               *mut Complex<f32>,
        workspace_in_bytes_on_host:   usize,
        info:                         *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
          handle,
          params,
          jobz,
          uplo,
          n,
          CUDA_C_32F,
          reinterpret_cast<void*>(A),
          lda,
          CUDA_R_32F,
          reinterpret_cast<void*>(W),
          CUDA_C_32F,
          reinterpret_cast<void*>(bufferOnDevice),
          workspaceInBytesOnDevice,
          reinterpret_cast<void*>(bufferOnHost),
          workspaceInBytesOnHost,
          info));
        */
}

#[cfg(USE_CUSOLVER_64_BIT)]
#[cfg(CUDART_VERSION)]
pub fn xsyevd_complex_double2(
        handle:                       CuSolverDnHandle,
        params:                       CuSolverDnParams,
        jobz:                         CuSolverEigMode,
        uplo:                         CuBlasFillMode,
        n:                            i64,
        A:                            *mut Complex<f64>,
        lda:                          i64,
        W:                            *mut f64,
        buffer_on_device:             *mut Complex<f64>,
        workspace_in_bytes_on_device: usize,
        buffer_on_host:               *mut Complex<f64>,
        workspace_in_bytes_on_host:   usize,
        info:                         *mut i32)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
          handle,
          params,
          jobz,
          uplo,
          n,
          CUDA_C_64F,
          reinterpret_cast<void*>(A),
          lda,
          CUDA_R_64F,
          reinterpret_cast<void*>(W),
          CUDA_C_64F,
          reinterpret_cast<void*>(bufferOnDevice),
          workspaceInBytesOnDevice,
          reinterpret_cast<void*>(bufferOnHost),
          workspaceInBytesOnHost,
          info));
        */
}
