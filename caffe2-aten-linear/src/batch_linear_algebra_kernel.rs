crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/BatchLinearAlgebraKernel.cpp]

/**
   | Computes the Cholesky decomposition of matrices
   | stored in `input`.
   |
   | This is an in-place routine and the content of
   | 'input' is overwritten with the result.
   |
   | Args:
   |
   | - `input` - [in] Input tensor for the Cholesky
   |                   decomposition
   |
   |              [out] Cholesky decomposition result
   |
   |  - `info` -  [out] Tensor filled with LAPACK
   |                    error codes, positive values
   |                    indicate that the matrix is
   |                    not positive definite.
   |
   |  - `upper` - controls whether the upper (true) or
   |     lower (false) triangular portion of `input` is
   |     used
   |
   |  For further details, please see the LAPACK
   |  documentation for POTRF.
   */
pub fn apply_cholesky<Scalar>(
        input: &Tensor,
        info:  &Tensor,
        upper: bool)  {

    todo!();
        /*
            #ifndef USE_LAPACK
      TORCH_CHECK(
          false,
          "Calling torch.linalg.cholesky on a CPU tensor requires compiling ",
          "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
    #else
      char uplo = upper ? 'U' : 'L';
      auto input_data = input.data_ptr<Scalar>();
      auto info_data = info.data_ptr<int>();
      auto input_matrix_stride = matrixStride(input);
      auto batch_size = batchCount(input);
      auto n = input.size(-2);
      auto lda = max<i64>(1, n);

      for (const auto i : irange(batch_size)) {
        Scalar* input_working_ptr = &input_data[i * input_matrix_stride];
        int* info_working_ptr = &info_data[i];
        lapackCholesky<Scalar>(uplo, n, input_working_ptr, lda, info_working_ptr);
      }
    #endif
        */
}

/// This is a type dispatching helper function for
/// 'apply_cholesky'
///
pub fn cholesky_kernel(
        input: &Tensor,
        infos: &Tensor,
        upper: bool)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "cholesky_cpu", [&]{
        apply_cholesky<Scalar>(input, infos, upper);
      });
        */
}

/**
  | Copies the lower (or upper) triangle
  | of the square matrix to the other half
  | and conjugates it.
  | 
  | This operation is performed in-place.
  |
  */
pub fn apply_reflect_conj_tri_single<Scalar>(
        self_:  *mut Scalar,
        n:      i64,
        stride: i64,
        upper:  bool)  {

    todo!();
        /*
            function<void(i64, i64)> loop = [](i64, i64){};
      if (upper) {
        loop = [&](i64 start, i64 end) {
          for (i64 i = start; i < end; i++) {
            for (i64 j = i + 1; j < n; j++) {
              self[i * stride + j] = conj_impl(self[j * stride + i]);
            }
          }
        };
      } else {
        loop = [&](i64 start, i64 end) {
          for (i64 i = start; i < end; i++) {
            for (i64 j = 0; j < i; j++) {
              self[i * stride + j] = conj_impl(self[j * stride + i]);
            }
          }
        };
      }
      // For small matrices OpenMP overhead is too large
      if (n < 256) {
        loop(0, n);
      } else {
        parallel_for(0, n, 0, loop);
      }
        */
}

/**
  | Computes the inverse of a symmetric
  | (Hermitian) positive-definite matrix
  | n-by-n matrix 'input' using the Cholesky
  | factorization
  | 
  | This is an in-place routine, content
  | of 'input' is overwritten. 'infos'
  | is an int Tensor containing error codes
  | for each matrix in the batched input.
  | 
  | For more information see LAPACK's documentation
  | for POTRI routine.
  |
  */
pub fn apply_cholesky_inverse<Scalar>(
        input: &mut Tensor,
        infos: &mut Tensor,
        upper: bool)  {

    todo!();
        /*
            #ifndef USE_LAPACK
      TORCH_CHECK(false, "cholesky_inverse: LAPACK library not found in compilation");
    #else
      char uplo = upper ? 'U' : 'L';

      auto input_data = input.data_ptr<Scalar>();
      auto infos_data = infos.data_ptr<int>();
      auto input_matrix_stride = matrixStride(input);
      auto batch_size = batchCount(input);
      auto n = input.size(-2);
      auto lda = max<i64>(1, n);

      for (i64 i = 0; i < batch_size; i++) {
        Scalar* input_working_ptr = &input_data[i * input_matrix_stride];
        int* info_working_ptr = &infos_data[i];
        lapackCholeskyInverse<Scalar>(uplo, n, input_working_ptr, lda, info_working_ptr);
        // LAPACK writes to only upper/lower part of the matrix leaving the other side unchanged
        apply_reflect_conj_tri_single<Scalar>(input_working_ptr, n, lda, upper);
      }
    #endif
        */
}

/**
  | This is a type dispatching helper function
  | for 'apply_cholesky_inverse'
  |
  */
pub fn cholesky_inverse_kernel_impl<'a>(
        result: &mut Tensor,
        infos:  &mut Tensor,
        upper:  bool) -> &'a mut Tensor {
    
    todo!();
        /*
            // This function calculates the inverse matrix in-place
      // result should be in column major order and contain matrices to invert
      // the content of result is overwritten by 'apply_cholesky_inverse'
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "cholesky_inverse_out_cpu", [&]{
        apply_cholesky_inverse<Scalar>(result, infos, upper);
      });
      return result;
        */
}

pub fn apply_eig<Scalar>(
        self_:        &Tensor,
        eigenvectors: bool,
        vals:         &mut Tensor,
        vecs:         &mut Tensor,
        info_ptr:     *mut i64)  {

    todo!();
        /*
            #ifndef USE_LAPACK
      TORCH_CHECK(false, "Calling torch.eig on a CPU tensor requires compiling ",
        "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
    #else
      using Value = typename scalar_Valueype<Scalar>::type;

      char jobvr = eigenvectors ? 'V' : 'N';
      i64 n = self.size(-1);
      auto self_data = self.data_ptr<Scalar>();

      auto vals_data = vals_.data_ptr<Scalar>();
      Scalar* wr = vals_data;

      Scalar* vecs_data = eigenvectors ? vecs_.data_ptr<Scalar>() : nullptr;
      int ldvr = eigenvectors ? n : 1;

      Tensor rwork;
      Value* rwork_data = nullptr;
      if (self.is_complex()) {
        ScalarType real_dtype = toValueType(typeMetaToScalarType(self.dtype()));
        rwork = empty({n*2}, self.options().dtype(real_dtype));
        rwork_data = rwork.data_ptr<Value>();
      }

      if (n > 0) {
        // call lapackEig once to get the optimal size for work data
        Scalar wkopt;
        int info;
        lapackEig<Scalar, Value>('N', jobvr, n, self_data, n, wr,
          nullptr, 1, vecs_data, ldvr, &wkopt, -1, rwork_data, &info);
        int lwork = max<int>(1, real_impl<Scalar, Value>(wkopt));

        // call again to do the actual work
        Tensor work = empty({lwork}, self.dtype());
        lapackEig<Scalar, Value>('N', jobvr, n, self_data, n, wr,
          nullptr, 1, vecs_data, ldvr, work.data_ptr<Scalar>(), lwork, rwork_data, &info);
        *info_ptr = info;
      }
    #endif
        */
}

pub fn eig_kernel_impl(
        self_:        &Tensor,
        eigenvectors: &mut bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            i64 n = self.size(-1);
      // lapackEig function expects the input to be column major, or stride {1, n},
      // so we must set the stride manually since the default stride for tensors is
      // row major, {n, 1}
      Tensor self_ = empty_strided(
          {n, n},
          {1, n},
          TensorOptions(self.dtype()));
      self_.copy_(self);

      auto options = self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      // the API is slightly different for the complex vs real case: if the input
      // is complex, eigenvals will be a vector of complex. If the input is real,
      // eigenvals will be a (n, 2) matrix containing the real and imaginary parts
      // in each column
      Tensor vals_;
      if (self.is_complex()) {
          vals_ = empty({n}, options);
      } else {
          vals_ = empty_strided({n, 2}, {1, n}, options);
      }
      Tensor vecs_ = eigenvectors
                     ? empty_strided({n, n}, {1, n}, options)
                     : Tensor();

      i64 info;
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "eig_cpu", [&]{
        apply_eig<Scalar>(self_, eigenvectors, vals_, vecs_, &info);
      });
      singleCheckErrors(info, "eig_cpu");
      return tuple<Tensor, Tensor>(vals_, vecs_);
        */
}


/**
  | Computes the eigenvalues and eigenvectors
  | of n-by-n matrix 'input'.
  | 
  | This is an in-place routine, content
  | of 'input', 'values', 'vectors' is
  | overwritten. 'infos' is an int Tensor
  | containing error codes for each matrix
  | in the batched input.
  | 
  | For more information see LAPACK's documentation
  | for GEEV routine.
  |
  */
pub fn apply_linalg_eig<Scalar>(
        values:               &mut Tensor,
        vectors:              &mut Tensor,
        input:                &mut Tensor,
        infos:                &mut Tensor,
        compute_eigenvectors: bool)  {

    todo!();
        /*
            #ifndef USE_LAPACK
      TORCH_CHECK(false, "Calling torch.linalg.eig on a CPU tensor requires compiling ",
        "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
    #else
      using Value = typename scalar_Valueype<Scalar>::type;

      char jobvr = compute_eigenvectors ? 'V' : 'N';
      char jobvl = 'N';  // only right eigenvectors are computed
      auto n = input.size(-1);
      auto lda = max<i64>(1, n);
      auto batch_size = batchCount(input);
      auto input_matrix_stride = matrixStride(input);
      auto values_stride = values.size(-1);
      auto input_data = input.data_ptr<Scalar>();
      auto values_data = values.data_ptr<Scalar>();
      auto infos_data = infos.data_ptr<int>();
      auto rvectors_data = compute_eigenvectors ? vectors.data_ptr<Scalar>() : nullptr;
      Scalar* lvectors_data = nullptr;  // only right eigenvectors are computed
      i64 ldvr = compute_eigenvectors ? lda : 1;
      i64 ldvl = 1;

      Tensor rwork;
      Value* rwork_data = nullptr;
      if (input.is_complex()) {
        ScalarType real_dtype = toValueType(input.scalar_type());
        rwork = empty({lda * 2}, input.options().dtype(real_dtype));
        rwork_data = rwork.data_ptr<Value>();
      }

      // call lapackEig once to get the optimal size for work data
      Scalar work_query;
      lapackEig<Scalar, Value>(jobvl, jobvr, n, input_data, lda, values_data,
        lvectors_data, ldvl, rvectors_data, ldvr, &work_query, -1, rwork_data, &infos_data[0]);

      int lwork = max<int>(1, static_cast<int>(real_impl<Scalar, Value>(work_query)));
      Tensor work = empty({lwork}, input.dtype());
      auto work_data = work.data_ptr<Scalar>();

      for (auto i = decltype(batch_size){0}; i < batch_size; i++) {
        Scalar* input_working_ptr = &input_data[i * input_matrix_stride];
        Scalar* values_working_ptr = &values_data[i * values_stride];
        Scalar* rvectors_working_ptr = compute_eigenvectors ? &rvectors_data[i * input_matrix_stride] : nullptr;
        int* info_working_ptr = &infos_data[i];
        lapackEig<Scalar, Value>(jobvl, jobvr, n, input_working_ptr, lda, values_working_ptr,
          lvectors_data, ldvl, rvectors_working_ptr, ldvr, work_data, lwork, rwork_data, info_working_ptr);
      }
    #endif
        */
}

/// This is a type dispatching helper function for
/// 'apply_linalg_eig'
///
pub fn linalg_eig_kernel(
        eigenvalues:          &mut Tensor,
        eigenvectors:         &mut Tensor,
        infos:                &mut Tensor,
        input:                &Tensor,
        compute_eigenvectors: bool)  {
    
    todo!();
        /*
            // This function calculates the non-symmetric eigendecomposition in-place
      // tensors should be in batched column major memory format
      // the content of eigenvalues, eigenvectors and infos is overwritten by 'apply_linalg_eig'

      // apply_linalg_eig modifies in-place provided input matrix, therefore we need a copy
      Tensor input_working_copy = empty(input.transpose(-2, -1).sizes(), input.options());
      input_working_copy.transpose_(-2, -1);  // make input_working_copy to have Fortran contiguous memory layout
      input_working_copy.copy_(input);

      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "linalg_eig_out_cpu", [&]{
        apply_linalg_eig<Scalar>(eigenvalues, eigenvectors, input_working_copy, infos, compute_eigenvectors);
      });
        */
}


/**
  | Computes eigenvalues and eigenvectors
  | of the input that is stored initially
  | in 'vectors'.
  | 
  | The computation is done in-place: 'vectors'
  | stores the input and will be overwritten,
  | 'values' should be an allocated empty
  | array. 'infos' is used to store information
  | for possible checks for error. 'upper'
  | controls the portion of input matrix
  | to consider in computations 'compute_eigenvectors'
  | controls whether eigenvectors should
  | be computed.
  | 
  | This function doesn't do any error checks
  | and it's assumed that every argument
  | is valid.
  |
  */
pub fn apply_lapack_eigh<Scalar>(
        values:               &mut Tensor,
        vectors:              &mut Tensor,
        infos:                &mut Tensor,
        upper:                bool,
        compute_eigenvectors: bool)  {

    todo!();
        /*
            #ifndef USE_LAPACK
      TORCH_CHECK(
          false,
          "Calling torch.linalg.eigh or eigvalsh on a CPU tensor requires compiling ",
          "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
    #else
      using Value = typename scalar_Valueype<Scalar>::type;

      char uplo = upper ? 'U' : 'L';
      char jobz = compute_eigenvectors ? 'V' : 'N';

      auto n = vectors.size(-1);
      auto lda = max<i64>(1, n);
      auto batch_size = batchCount(vectors);

      auto vectors_stride = matrixStride(vectors);
      auto values_stride = values.size(-1);

      auto vectors_data = vectors.data_ptr<Scalar>();
      auto values_data = values.data_ptr<Value>();
      auto infos_data = infos.data_ptr<int>();

      // Using 'int' instead of i32 or i64 is consistent with the current LAPACK interface
      // It really should be changed in the future to something like lapack_int that depends on the specific LAPACK library that is linked
      // or switch to supporting only 64-bit indexing by default.
      int lwork = -1;
      int lrwork = -1;
      int liwork = -1;
      Scalar lwork_query;
      Value rwork_query;
      int iwork_query;

      // call lapackSyevd once to get the optimal size for work data
      lapackSyevd<Scalar, Value>(jobz, uplo, n, vectors_data, lda, values_data,
        &lwork_query, lwork, &rwork_query, lrwork, &iwork_query, liwork, infos_data);

      lwork = max<int>(1, real_impl<Scalar, Value>(lwork_query));
      Tensor work = empty({lwork}, vectors.options());
      auto work_data = work.data_ptr<Scalar>();

      liwork = max<int>(1, iwork_query);
      Tensor iwork = empty({liwork}, vectors.options().dtype(kInt));
      auto iwork_data = iwork.data_ptr<int>();

      Tensor rwork;
      Value* rwork_data = nullptr;
      if (vectors.is_complex()) {
        lrwork = max<int>(1, rwork_query);
        rwork = empty({lrwork}, values.options());
        rwork_data = rwork.data_ptr<Value>();
      }

      // Now call lapackSyevd for each matrix in the batched input
      for (const auto i : irange(batch_size)) {
        Scalar* vectors_working_ptr = &vectors_data[i * vectors_stride];
        Value* values_working_ptr = &values_data[i * values_stride];
        int* info_working_ptr = &infos_data[i];
        lapackSyevd<Scalar, Value>(jobz, uplo, n, vectors_working_ptr, lda, values_working_ptr,
          work_data, lwork, rwork_data, lrwork, iwork_data, liwork, info_working_ptr);
        // The current behaviour for Linear Algebra functions to raise an error if something goes wrong
        // or input doesn't satisfy some requirement
        // therefore return early since further computations will be wasted anyway
        if (*info_working_ptr != 0) {
          return;
        }
      }
    #endif
        */
}

/// This is a type dispatching helper function for
/// 'apply_lapack_eigh'
///
pub fn linalg_eigh_kernel(
        eigenvalues:          &mut Tensor,
        eigenvectors:         &mut Tensor,
        infos:                &mut Tensor,
        upper:                bool,
        compute_eigenvectors: bool)  {
    
    todo!();
        /*
            // This function calculates the symmetric/hermitian eigendecomposition
      // in-place tensors should be in batched column major memory format the
      // content of eigenvalues, eigenvectors and infos is overwritten by
      // 'apply_lapack_eigh'
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
          eigenvectors.scalar_type(), "linalg_eigh_cpu", [&] {
            apply_lapack_eigh<Scalar>(
                eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
          });
        */
}

/**
   |  The geqrf function computes the QR decomposition of matrices stored in `input`.
   |  However, rather than producing a Q matrix directly, it produces a sequence of
   |  elementary reflectors which may later be composed to construct Q - for example
   |  with the orgqr or ormqr functions.
   |
   |  Args:
   |  * `input` - [in] Input tensor for QR decomposition
   |              [out] QR decomposition result which contains:
   |              i)  The elements of R, on and above the diagonal.
   |              ii) Directions of the reflectors implicitly defining Q.
   |             Tensor with the directions of the elementary reflectors below the diagonal,
   |              it will be overwritten with the result
   |  * `tau` - [out] Tensor which will contain the magnitudes of the reflectors
   |            implicitly defining Q.
   |
   |  For further details, please see the LAPACK documentation for GEQRF.
   */
pub fn apply_geqrf<Scalar>(
        input: &Tensor,
        tau:   &Tensor)  {

    todo!();
        /*
            #ifndef USE_LAPACK
      TORCH_CHECK(
          false,
          "Calling torch.geqrf on a CPU tensor requires compiling ",
          "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
    #else
      using Value = typename scalar_Valueype<Scalar>::type;
      auto input_data = input.data_ptr<Scalar>();
      auto tau_data = tau.data_ptr<Scalar>();
      auto input_matrix_stride = matrixStride(input);
      auto tau_stride = tau.size(-1);
      auto batch_size = batchCount(input);
      auto m = input.size(-2);
      auto n = input.size(-1);
      auto lda = max<int>(1, m);

      int info;
      // Run once, first to get the optimum work size.
      // Since we deal with batches of matrices with the same dimensions, doing this outside
      // the loop saves (batch_size - 1) workspace queries which would provide the same result
      // and (batch_size - 1) calls to allocate and deallocate workspace using empty()
      int lwork = -1;
      Scalar wkopt;
      lapackGeqrf<Scalar>(m, n, input_data, lda, tau_data, &wkopt, lwork, &info);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);

      // if lwork is less than 'n' then a warning is printed:
      // Intel MKL ERROR: Parameter 7 was incorrect on entry to SGEQRF.
      lwork = max<int>(max<int>(1, n), real_impl<Scalar, Value>(wkopt));
      Tensor work = empty({lwork}, input.options());

      for (const auto i : irange(batch_size)) {
        Scalar* input_working_ptr = &input_data[i * input_matrix_stride];
        Scalar* tau_working_ptr = &tau_data[i * tau_stride];

        // now compute the actual QR and tau
        lapackGeqrf<Scalar>(m, n, input_working_ptr, lda, tau_working_ptr, work.data_ptr<Scalar>(), lwork, &info);

        // info from lapackGeqrf only reports if the i-th parameter is wrong
        // so we don't need to check it all the time
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
      }
    #endif
        */
}

/// This is a type dispatching helper function for
/// 'apply_geqrf'
///
pub fn geqrf_kernel(
        input: &Tensor,
        tau:   &Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "geqrf_cpu", [&]{
        apply_geqrf<Scalar>(input, tau);
      });
        */
}

/**
   |  The orgqr function allows reconstruction of an orthogonal (or unitary) matrix Q,
   |  from a sequence of elementary reflectors, such as produced by the geqrf function.
   |
   |  Args:
   |  * `self` - Tensor with the directions of the elementary reflectors below the diagonal,
   |              it will be overwritten with the result
   |  * `tau` - Tensor containing the magnitudes of the elementary reflectors
   |
   |  For further details, please see the LAPACK documentation for ORGQR and UNGQR.
   */
#[inline] pub fn apply_orgqr<Scalar>(
        self_: &mut Tensor,
        tau:   &Tensor)  {

    todo!();
        /*
            #ifndef USE_LAPACK
      TORCH_CHECK(false, "Calling torch.orgqr on a CPU tensor requires compiling ",
        "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
    #else
      // Some LAPACK implementations might not work well with empty matrices:
      // workspace query might return lwork as 0, which is not allowed (requirement is lwork >= 1)
      // We don't need to do any calculations in this case, so let's return early
      if (self.numel() == 0) {
        return;
      }

      using Value = typename scalar_Valueype<Scalar>::type;
      auto self_data = self.data_ptr<Scalar>();
      auto tau_data = tau.data_ptr<Scalar>();
      auto self_matrix_stride = matrixStride(self);
      auto tau_stride = tau.size(-1);
      auto batch_size = batchCount(self);
      auto m = self.size(-2);
      auto n = self.size(-1);
      auto k = tau.size(-1);
      auto lda = max<i64>(1, m);
      int info;

      // LAPACK's requirement
      TORCH_INTERNAL_ASSERT(m >= n);
      TORCH_INTERNAL_ASSERT(n >= k);

      // Run once, first to get the optimum work size.
      // Since we deal with batches of matrices with the same dimensions, doing this outside
      // the loop saves (batch_size - 1) workspace queries which would provide the same result
      // and (batch_size - 1) calls to allocate and deallocate workspace using empty()
      int lwork = -1;
      Scalar wkopt;
      lapackOrgqr<Scalar>(m, n, k, self_data, lda, tau_data, &wkopt, lwork, &info);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
      lwork = max<int>(1, real_impl<Scalar, Value>(wkopt));
      Tensor work = empty({lwork}, self.options());

      for (i64 i = 0; i < batch_size; i++) {
        Scalar* self_working_ptr = &self_data[i * self_matrix_stride];
        Scalar* tau_working_ptr = &tau_data[i * tau_stride];

        // now compute the actual Q
        lapackOrgqr<Scalar>(m, n, k, self_working_ptr, lda, tau_working_ptr, work.data_ptr<Scalar>(), lwork, &info);

        // info from lapackOrgqr only reports if the i-th parameter is wrong
        // so we don't need to check it all the time
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
      }
    #endif
        */
}

/// This is a type dispatching helper function for 'apply_orgqr'
///
pub fn orgqr_kernel_impl<'a>(
        result: &mut Tensor,
        tau:    &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "orgqr_cpu", [&]{
        apply_orgqr<Scalar>(result, tau);
      });
      return result;
        */
}



// we use `enum class LapackLstsqDriverType` as keys in an unordered_map.
// Clang5 and Gcc5 do not support hash for enum classes, hence
// we provide our own hash function.
pub struct LapackLstsqDriverTypeHash {

}

impl LapackLstsqDriverTypeHash {
    
    pub fn invoke(&self, driver_type: &LapackLstsqDriverType) -> usize {
        
        todo!();
        /*
            return static_cast<usize>(driver_type);
        */
    }
}

/**
  |  Solves a least squares problem. That is minimizing ||B - A X||.
  |
  |  Input args:
  |  * 'input' - Tensor containing batches of m-by-n matrix A.
  |  * 'other' - Tensor containing batches of max(m, n)-by-nrhs matrix B.
  |  * 'cond' - relative tolerance for determining rank of A.
  |  * 'driver' - the name of the LAPACK driver that is used to compute the solution.
  |  Output args (modified in-place):
  |  * 'solution' - Tensor to store the solution matrix X.
  |  * 'residuals' - Tensor to store values of ||B - A X||.
  |  * 'rank' - Tensor to store the rank of A.
  |  * 'singular_values' - Tensor to store the singular values of A.
  |  * 'infos' - Tensor to store error codes of linear algebra math library.
  |
  |  For further details, please see the LAPACK documentation for GELS/GELSY/GELSS/GELSD routines.
*/
pub fn apply_lstsq<Scalar>(
        A:               &Tensor,
        B:               &mut Tensor,
        rank:            &mut Tensor,
        singular_values: &mut Tensor,
        infos:           &mut Tensor,
        rcond:           f64,
        driver_type:     LapackLstsqDriverType)  {

    todo!();
        /*
            #ifndef USE_LAPACK
      TORCH_CHECK(
          false,
          "Calling torch.linalg.lstsq on a CPU tensor requires compiling ",
          "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
    #else
      using Value = typename scalar_Valueype<Scalar>::type;
      using driver_t = native::LapackLstsqDriverType;

      auto lapack_func = lapackLstsq<driver_t::Gelsd, Scalar, Value>;
      static auto driver_type_to_func
        = unordered_map<driver_t, decltype(lapack_func), LapackLstsqDriverTypeHash>({
        {driver_t::Gels, lapackLstsq<driver_t::Gels, Scalar, Value>},
        {driver_t::Gelsy, lapackLstsq<driver_t::Gelsy, Scalar, Value>},
        {driver_t::Gelsd, lapackLstsq<driver_t::Gelsd, Scalar, Value>},
        {driver_t::Gelss, lapackLstsq<driver_t::Gelss, Scalar, Value>}
      });
      lapack_func = driver_type_to_func[driver_type];

      char trans = 'N';

      auto A_data = A.data_ptr<Scalar>();
      auto B_data = B.data_ptr<Scalar>();
      auto m = A.size(-2);
      auto n = A.size(-1);
      auto nrhs = B.size(-1);
      auto lda = max<i64>(1, m);
      auto ldb = max<i64>(1, max(m, n));
      auto infos_data = infos.data_ptr<int>();

      // only 'gels' driver does not compute the rank
      int rank_32;
      i64* rank_data;
      i64* rank_working_ptr = nullptr;
      if (driver_t::Gels != driver_type) {
        rank_data = rank.data_ptr<i64>();
        rank_working_ptr = rank_data;
      }

      // 'gelsd' and 'gelss' are SVD-based algorithms
      // so we can get singular values
      Value* s_data;
      Value* s_working_ptr = nullptr;
      i64 s_stride;
      if (driver_t::Gelsd == driver_type || driver_t::Gelss == driver_type) {
        s_data = singular_values.data_ptr<Value>();
        s_working_ptr = s_data;
        s_stride = singular_values.size(-1);
      }

      // 'jpvt' workspace array is used only for 'gelsy' which uses QR factorization with column pivoting
      Tensor jpvt;
      int* jpvt_data = nullptr;
      if (driver_t::Gelsy == driver_type) {
        jpvt = empty({max<i64>(1, n)}, A.options().dtype(kInt));
        jpvt_data = jpvt.data_ptr<int>();
      }

      // Run once the driver, first to get the optimal workspace size
      int lwork = -1; // default value to decide the opt size for workspace arrays
      Scalar work_opt;
      Value rwork_opt;
      int iwork_opt;
      lapack_func(trans, m, n, nrhs,
        A_data, lda,
        B_data, ldb,
        &work_opt, lwork,
        infos_data,
        jpvt_data,
        static_cast<Value>(rcond),
        &rank_32,
        &rwork_opt,
        s_working_ptr,
        &iwork_opt);

      lwork = max<int>(1, real_impl<Scalar, Value>(work_opt));
      Tensor work = empty({lwork}, A.options());
      Scalar* work_data = work.data_ptr<Scalar>();

      // 'rwork' only used for complex inputs and 'gelsy', 'gelsd' and 'gelss' drivers
      Tensor rwork;
      Value* rwork_data;
      if (A.is_complex() && driver_t::Gels != driver_type) {
        i64 rwork_len;
        switch (driver_type) {
          case driver_t::Gelsy:
            rwork_len = max<i64>(1, 2 * n);
            break;
          case driver_t::Gelss:
            rwork_len = max<i64>(1, 5 * min(m, n));
            break;
          // case driver_t::Gelsd:
          default:
            rwork_len = max<i64>(1, rwork_opt);
        }
        rwork = empty({rwork_len}, A.options().dtype(toValueType(A.scalar_type())));
        rwork_data = rwork.data_ptr<Value>();
      }

      // 'iwork' workspace array is relevant only for 'gelsd'
      Tensor iwork;
      int* iwork_data;
      if (driver_t::Gelsd == driver_type) {
        iwork = empty({max<int>(1, iwork_opt)}, A.options().dtype(kInt));
        iwork_data = iwork.data_ptr<int>();
      }

      native::batch_iterator_with_broadcasting<Scalar>(A, B,
        [&](Scalar* A_working_ptr, Scalar* B_working_ptr, i64 A_linear_batch_idx) {
          rank_working_ptr = rank_working_ptr ? &rank_data[A_linear_batch_idx] : nullptr;
          s_working_ptr = s_working_ptr ? &s_data[A_linear_batch_idx * s_stride] : nullptr;
          int* infos_working_ptr = &infos_data[A_linear_batch_idx];

          lapack_func(trans, m, n, nrhs,
            A_working_ptr, lda,
            B_working_ptr, ldb,
            work_data, lwork,
            infos_working_ptr,
            jpvt_data,
            static_cast<Value>(rcond),
            &rank_32,
            rwork_data,
            s_working_ptr,
            iwork_data);

          // we want the output `rank` Tensor to be of type i64,
          // however LAPACK accepts int. That is why we use an integer
          // variable that then gets promoted and written into `rank`.
          // We use this approach over a tensor cast for better performance.
          if (rank_working_ptr) {
            *rank_working_ptr = static_cast<i64>(rank_32);
          }
        }
      );
    #endif
        */
}

/// This is a type and driver dispatching helper
/// function for 'apply_lstsq'
///
pub fn lstsq_kernel(
        a:               &Tensor,
        b:               &mut Tensor,
        rank:            &mut Tensor,
        singular_values: &mut Tensor,
        infos:           &mut Tensor,
        rcond:           f64,
        driver_name:     String)  {
    
    todo!();
        /*
            static auto driver_string_to_type = unordered_map<string_view, LapackLstsqDriverType>({
        {"gels", native::LapackLstsqDriverType::Gels},
        {"gelsy", native::LapackLstsqDriverType::Gelsy},
        {"gelsd", native::LapackLstsqDriverType::Gelsd},
        {"gelss", native::LapackLstsqDriverType::Gelss}
      });
      auto driver_type = driver_string_to_type[driver_name];

      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(a.scalar_type(), "linalg_lstsq_cpu", [&]{
        apply_lstsq<Scalar>(a, b, rank, singular_values, infos, rcond, driver_type);
      });
        */
}

/**
  |  The ormqr function multiplies Q with another matrix from a sequence of
  |  elementary reflectors, such as is produced by the geqrf function.
  |
  |  Args:
  |  * `input`     - Tensor with elementary reflectors below the diagonal,
  |                  encoding the matrix Q.
  |  * `tau`       - Tensor containing the magnitudes of the elementary
  |                  reflectors.
  |  * `other`     - [in] Tensor containing the matrix to be multiplied.
  |                  [out] result of the matrix multiplication with Q.
  |  * `left`      - bool, determining whether `other` is left- or right-multiplied with Q.
  |  * `transpose` - bool, determining whether to transpose (or conjugate transpose) Q before multiplying.
  |
  |  For further details, please see the LAPACK documentation.
*/
pub fn apply_ormqr<Scalar>(
        input:     &Tensor,
        tau:       &Tensor,
        other:     &Tensor,
        left:      bool,
        transpose: bool)  {

    todo!();
        /*
            #ifndef USE_LAPACK
      TORCH_CHECK(false, "Calling torch.ormqr on a CPU tensor requires compiling ",
        "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
    #else
      using Value = typename scalar_Valueype<Scalar>::type;

      char side = left ? 'L' : 'R';
      char trans = transpose ? (input.is_complex() ? 'C' : 'T') : 'N';

      auto input_data = input.data_ptr<Scalar>();
      auto tau_data = tau.data_ptr<Scalar>();
      auto other_data = other.data_ptr<Scalar>();

      auto input_matrix_stride = matrixStride(input);
      auto other_matrix_stride = matrixStride(other);
      auto tau_stride = tau.size(-1);
      auto batch_size = batchCount(input);
      auto m = other.size(-2);
      auto n = other.size(-1);
      auto k = tau.size(-1);
      auto lda = max<i64>(1, left ? m : n);
      auto ldc = max<i64>(1, m);
      int info = 0;

      // LAPACK's requirement
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY((left ? m : n) >= k);

      // Query for the optimal size of the workspace tensor
      int lwork = -1;
      Scalar wkopt;
      lapackOrmqr<Scalar>(side, trans, m, n, k, input_data, lda, tau_data, other_data, ldc, &wkopt, lwork, &info);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
      lwork = max<int>(1, real_impl<Scalar, Value>(wkopt));
      Tensor work = empty({lwork}, input.options());

      for (const auto i : irange(batch_size)) {
        Scalar* input_working_ptr = &input_data[i * input_matrix_stride];
        Scalar* other_working_ptr = &other_data[i * other_matrix_stride];
        Scalar* tau_working_ptr = &tau_data[i * tau_stride];

        // now compute the actual result
        lapackOrmqr<Scalar>(
            side, trans, m, n, k,
            input_working_ptr, lda,
            tau_working_ptr,
            other_working_ptr, ldc,
            work.data_ptr<Scalar>(), lwork, &info);

        // info from lapackOrmqr only reports if the i-th parameter is wrong
        // so we don't need to check it all the time
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
      }
    #endif
        */
}

/// This is a type dispatching helper function for
/// 'apply_ormqr'
///
pub fn ormqr_kernel(
        input:     &Tensor,
        tau:       &Tensor,
        other:     &Tensor,
        left:      bool,
        transpose: bool)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "ormqr_cpu", [&]{
        apply_ormqr<Scalar>(input, tau, other, left, transpose);
      });
        */
}

/*
  |Solves the matrix equation op(A) X = B
  |X and B are n-by-nrhs matrices, A is a unit, or non-unit, upper or lower triangular matrix
  |and op(A) is one of op(A) = A or op(A) = A^T or op(A) = A^H.
  |This is an in-place routine, content of 'B' is overwritten.
  |'upper' controls the portion of input matrix to consider in computations,
  |'transpose' if true then op(A) = A^T,
  |'unitriangular' if true then the diagonal elements of A are assumed to be 1
  |and the actual diagonal values are not used.
  |'infos' is an int Tensor containing error codes for each matrix in the batched input.
  |For more information see LAPACK's documentation for TRTRS routine.
*/
pub fn apply_triangular_solve<Scalar>(
        A:                   &mut Tensor,
        B:                   &mut Tensor,
        infos:               &mut Tensor,
        upper:               bool,
        transpose:           bool,
        conjugate_transpose: bool,
        unitriangular:       bool)  {

    todo!();
        /*
            #ifndef USE_LAPACK
      TORCH_CHECK(
          false,
          "Calling torch.triangular_solve on a CPU tensor requires compiling ",
          "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
    #else
      char uplo = upper ? 'U' : 'L';
      char trans = transpose ? 'T' : 'N';
      trans = conjugate_transpose ? 'C' : trans;
      char diag = unitriangular ? 'U' : 'N';

      auto A_data = A.data_ptr<Scalar>();
      auto B_data = B.data_ptr<Scalar>();
      auto A_mat_stride = matrixStride(A);
      auto B_mat_stride = matrixStride(B);
      auto batch_size = batchCount(A);
      auto n = A.size(-2);
      auto nrhs = B.size(-1);
      auto lda = max<i64>(1, n);
      auto infos_data = infos.data_ptr<int>();

      for (const auto i : irange(batch_size)) {
        Scalar* A_working_ptr = &A_data[i * A_mat_stride];
        Scalar* B_working_ptr = &B_data[i * B_mat_stride];
        int* info_working_ptr = &infos_data[i];
        lapackTriangularSolve<Scalar>(uplo, trans, diag, n, nrhs, A_working_ptr, lda, B_working_ptr, lda, info_working_ptr);
        // The current behaviour for linear algebra functions to raise an error if something goes wrong
        // or input doesn't satisfy some requirement
        // therefore return early since further computations will be wasted anyway
        if (*info_working_ptr != 0) {
          return;
        }
      }
    #endif
        */
}

pub fn triangular_solve_kernel(
        A:                   &mut Tensor,
        B:                   &mut Tensor,
        infos:               &mut Tensor,
        upper:               bool,
        transpose:           bool,
        conjugate_transpose: bool,
        unitriangular:       bool)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "triangular_solve_cpu", [&]{
        apply_triangular_solve<Scalar>(A, B, infos, upper, transpose, conjugate_transpose, unitriangular);
      });
        */
}

/*
  |  Computes the LU decomposition of a m×n matrix or
  |  batch of matrices in 'input' tensor.
  |
  |  This is an in-place routine, content of 'input',
  |  'pivots', and 'infos' is overwritten.
  |
  |  Args:
  |  * `input` - [in] the input matrix for LU decomposition
  |              [out] the LU decomposition
  |  * `pivots` - [out] the pivot indices
  |  * `infos` - [out] error codes, positive values indicate singular matrices
  |  * `compute_pivots` - should always be true (can be false only for CUDA)
  |
  |  For further details, please see the LAPACK
  |  documentation for GETRF.
  |
*/
pub fn apply_lu<Scalar>(
        input:          &Tensor,
        pivots:         &Tensor,
        infos:          &Tensor,
        compute_pivots: bool)  {

    todo!();
        /*
            #ifndef USE_LAPACK
      TORCH_CHECK(
          false,
          "Calling torch.lu on a CPU tensor requires compiling ",
          "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
    #else
      TORCH_CHECK(compute_pivots, "lu without pivoting is not implemented on the CPU");

      auto input_data = input.data_ptr<Scalar>();
      auto pivots_data = pivots.data_ptr<int>();
      auto infos_data = infos.data_ptr<int>();
      auto input_matrix_stride = matrixStride(input);
      auto pivots_stride = pivots.size(-1);
      auto batch_size = batchCount(input);
      auto m = input.size(-2);
      auto n = input.size(-1);
      auto leading_dimension = max<i64>(1, m);

      for (const auto i : irange(batch_size)) {
        Scalar* input_working_ptr = &input_data[i * input_matrix_stride];
        int* pivots_working_ptr = &pivots_data[i * pivots_stride];
        int* infos_working_ptr = &infos_data[i];
        lapackLu<Scalar>(m, n, input_working_ptr, leading_dimension, pivots_working_ptr, infos_working_ptr);
      }
    #endif
        */
}

/// This is a type dispatching helper function for
/// 'apply_lu'
///
pub fn lu_kernel(
        input:          &Tensor,
        pivots:         &Tensor,
        infos:          &Tensor,
        compute_pivots: bool)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "lu_cpu", [&]{
        apply_lu<Scalar>(input, pivots, infos, compute_pivots);
      });
        */
}

/*
  |  Solves the matrix equation A X = B
  |
  |  X and B are n-by-nrhs matrices, A is represented
  |  using the LU factorization.
  |
  |  This is an in-place routine, content of `b` is
  |  overwritten.
  |
  |  Args:
  |  * `b` -  [in] the right hand side matrix B
  |           [out] the solution matrix X
  |  * `lu` - [in] the LU factorization of matrix A (see _lu_with_info)
  |  * `pivots` - [in] the pivot indices (see _lu_with_info)
  |
  |  For further details, please see the LAPACK
  |  documentation for GETRS.
*/
pub fn apply_lu_solve<Scalar>(
        b:      &Tensor,
        lu:     &Tensor,
        pivots: &Tensor)  {

    todo!();
        /*
            #ifndef USE_LAPACK
      TORCH_CHECK(
          false,
          "Calling torch.lu_solve on a CPU tensor requires compiling ",
          "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
    #else
      char trans = 'N';
      auto b_data = b.data_ptr<Scalar>();
      auto lu_data = lu.data_ptr<Scalar>();
      auto pivots_data = pivots.data_ptr<int>();
      auto b_stride = matrixStride(b);
      auto lu_stride = matrixStride(lu);
      auto pivots_stride = pivots.size(-1);
      auto batch_size = batchCount(b);

      auto n = lu.size(-2);
      auto nrhs = b.size(-1);
      auto leading_dimension = max<i64>(1, n);

      int info = 0;
      for (const auto i : irange(batch_size)) {
        Scalar* b_working_ptr = &b_data[i * b_stride];
        Scalar* lu_working_ptr = &lu_data[i * lu_stride];
        int* pivots_working_ptr = &pivots_data[i * pivots_stride];

        lapackLuSolve<Scalar>(trans, n, nrhs, lu_working_ptr, leading_dimension, pivots_working_ptr,
                                b_working_ptr, leading_dimension, &info);

        // info from lapackLuSolve only reports if the i-th parameter is wrong
        // so we don't need to check it all the time
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
      }
    #endif
        */
}

/// This is a type dispatching helper function for
/// 'apply_lu_solve'
///
pub fn lu_solve_kernel(
        b:      &Tensor,
        lu:     &Tensor,
        pivots: &Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(b.scalar_type(), "lu_solve_cpu", [&]{
        apply_lu_solve<Scalar>(b, lu, pivots);
      });
        */
}

register_avx_dispatch!{cholesky_inverse_stub  , &cholesky_inverse_kernel_impl}
register_avx_dispatch!{cholesky_stub          , &cholesky_kernel}
register_avx_dispatch!{eig_stub               , &eig_kernel_impl}
register_avx_dispatch!{geqrf_stub             , &geqrf_kernel}
register_avx_dispatch!{linalg_eig_stub        , &linalg_eig_kernel}
register_avx_dispatch!{linalg_eigh_stub       , &linalg_eigh_kernel}
register_avx_dispatch!{lstsq_stub             , &lstsq_kernel}
register_avx_dispatch!{lu_solve_stub          , &lu_solve_kernel}
register_avx_dispatch!{lu_stub                , &lu_kernel}
register_avx_dispatch!{orgqr_stub             , &orgqr_kernel_impl}
register_avx_dispatch!{ormqr_stub             , &ormqr_kernel}
register_avx_dispatch!{triangular_solve_stub  , &triangular_solve_kernel}
register_vsx_dispatch!{cholesky_inverse_stub  , &cholesky_inverse_kernel_impl}
register_vsx_dispatch!{cholesky_stub          , &cholesky_kernel}
register_vsx_dispatch!{eig_stub               , &eig_kernel_impl}
register_vsx_dispatch!{geqrf_stub             , &geqrf_kernel}
register_vsx_dispatch!{linalg_eig_stub        , &linalg_eig_kernel}
register_vsx_dispatch!{linalg_eigh_stub       , &linalg_eigh_kernel}
register_vsx_dispatch!{lstsq_stub             , &lstsq_kernel}
register_vsx_dispatch!{lu_solve_stub          , &lu_solve_kernel}
register_vsx_dispatch!{lu_stub                , &lu_kernel}
register_vsx_dispatch!{orgqr_stub             , &orgqr_kernel_impl}
register_vsx_dispatch!{ormqr_stub             , &ormqr_kernel}
register_vsx_dispatch!{triangular_solve_stub  , &triangular_solve_kernel}

register_avx2_dispatch!{cholesky_inverse_stub , &cholesky_inverse_kernel_impl}
register_avx2_dispatch!{cholesky_stub         , &cholesky_kernel}
register_avx2_dispatch!{eig_stub              , &eig_kernel_impl}
register_avx2_dispatch!{geqrf_stub            , &geqrf_kernel}
register_avx2_dispatch!{linalg_eig_stub       , &linalg_eig_kernel}
register_avx2_dispatch!{linalg_eigh_stub      , &linalg_eigh_kernel}
register_avx2_dispatch!{lstsq_stub            , &lstsq_kernel}
register_avx2_dispatch!{lu_solve_stub         , &lu_solve_kernel}
register_avx2_dispatch!{lu_stub               , &lu_kernel}
register_avx2_dispatch!{orgqr_stub            , &orgqr_kernel_impl}
register_avx2_dispatch!{ormqr_stub            , &ormqr_kernel}
register_avx2_dispatch!{triangular_solve_stub , &triangular_solve_kernel}

register_arch_dispatch!{linalg_eigh_stub      , DEFAULT , &linalg_eigh_kernel}
register_arch_dispatch!{cholesky_stub         , DEFAULT , &cholesky_kernel}
register_arch_dispatch!{cholesky_inverse_stub , DEFAULT , &cholesky_inverse_kernel_impl}
register_arch_dispatch!{eig_stub              , DEFAULT , &eig_kernel_impl}
register_arch_dispatch!{linalg_eig_stub       , DEFAULT , &linalg_eig_kernel}
register_arch_dispatch!{geqrf_stub            , DEFAULT , &geqrf_kernel}
register_arch_dispatch!{orgqr_stub            , DEFAULT , &orgqr_kernel_impl}
register_arch_dispatch!{ormqr_stub            , DEFAULT , &ormqr_kernel}
register_arch_dispatch!{lstsq_stub            , DEFAULT , &lstsq_kernel}
register_arch_dispatch!{triangular_solve_stub , DEFAULT , &triangular_solve_kernel}
register_arch_dispatch!{lu_stub               , DEFAULT , &lu_kernel}
register_arch_dispatch!{lu_solve_stub         , DEFAULT , &lu_solve_kernel}
