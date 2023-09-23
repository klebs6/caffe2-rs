crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/LinearAlgebraUtils.h]

/**
  | Clones a Tensor so that the following
  | conditions hold:
  | 
  | If we think of a Tensor of having size
  | (B, M, N), where B is any number of batch
  | dimensions, then:
  | 
  | - Each (M, N) matrix is in column major
  | form
  | 
  | - Let Tensor P have size (B, M, N) and Q
  | have size (B, M', N').
  | 
  | Then when laid out in memory, the M by
  | N matrix starting at
  | 
  | P.data_ptr()[b * M * N] is of the same
  | corresponding batch as the M' by N' matrix
  | starting at Q.data_ptr()[b * M' * N'].
  |
  */
#[inline] pub fn clone_batched_column_major(src: &Tensor) -> Tensor {
    
    todo!();
        /*
            // If src is already in batched column major format, then
      // this will be efficient (no reordering of the data will occur)
      // because the first transpose will make the tensor contiguous,
      // and cloning a contiguous tensor is fast.
      auto result = src.transpose(-2, -1).clone(MemoryFormat::Contiguous);
      result.transpose_(-2, -1);
      return result;
        */
}

/**
  | This method is designed to be a faster
  | alternative to `cloneBatchedColumnMajor`
  | with some additional features, namely:
  | 1. It uses `copy` instead of `clone`
  | which could be much faster. 2. `nrows`
  | parameter used to create inputs with
  | the number of rows larger than the original
  | input, which is required for some LAPACK/MAGMA
  | methods. 3. `desired_batch_size`
  | is used to create copies with the batch
  | size which is either the original batch
  | size of the input, or its larger broadcasted
  | shape.
  |
  */
#[inline] pub fn copy_batched_column_major(
        src:                 &Tensor,
        nrows:               i64,
        desired_batch_sizes: Option<&[i32]>) -> Tensor {

    let nrows: i64 = nrows.unwrap_or(-1);
    let desired_batch_sizes: Option<&[i32]> = desired_batch_sizes.unwrap_or(nullopt);

    todo!();
        /*
            nrows = (nrows == -1) ? src.size(-2) : nrows;
      auto copy_sizes = desired_batch_sizes.has_value()
        ? desired_batch_sizes.value().vec()
        : IntArrayRef(src.sizes().data(), src.dim() - 2).vec();
      copy_sizes.insert(copy_sizes.end(), {nrows, src.size(-1)});
      auto copy_strides = defaultStrides(copy_sizes);
      copy_strides[src.dim() - 2] = 1;
      copy_strides[src.dim() - 1] = nrows;
      auto copy = empty_strided(copy_sizes, copy_strides, src.options());
      copy.narrow(-2, 0, src.size(-2)).copy_(src);
      return copy;
        */
}

/**
  | Given batches of matrices with arbitrary
  | batch dim, computes the number of batches.
  |
  */
#[inline] pub fn batch_count(batched_matrices: &Tensor) -> i64 {
    
    todo!();
        /*
            i64 result = 1;
      for (i64 i = 0; i < batched_matrices.ndimension() - 2; i++) {
        result *= batched_matrices.size(i);
      }
      return result;
        */
}

/**
  | Computes the number of elements of a
  | matrix in a batched matrix tensor
  |
  */
#[inline] pub fn matrix_stride(batched_matrices: &Tensor) -> i64 {
    
    todo!();
        /*
            return batched_matrices.size(-1) * batched_matrices.size(-2);
        */
}

/**
  | This function is designed to be used with
  | linear algebra methods that minimize
  | L(ax - b) = 0, where L is generally the identity map
  | (`solve`, for example) or the L2 norm
  | (`lstsq`).
  |
  | It is expected that `a` and `b` are contiguous
  | tensors of column-major matrices (so that
  | a.view({-1, a.size(-2), a.size(-1)}) succeeds,
  | same for `b`), with the following additional
  | properties:
  |
  | 1. a.dim() == b.dim()
  | 2. a.shape[:-2] broadcasts over b.shape[:-2]
  | 3. a.size(i) <= b.size(i) for i=0,..., a.dim() - 3
  | (only for batch dimensions)
  |
  | MAGMA/LAPACK modify tensor `a` in-place, and
  | the main goal of this method is to be memory
  | efficient, which means that if there exists an
  | index i such that a.shape[i] < b.shape[i], 0 <=
  | i <= a.dim() - 3, then instead of materializing
  | copies of `a` in the broadcasted shape, we keep
  | a buffer copy of `a` along with flags that
  | check whether specific batch dimension indices
  | for `a` were already accessed.
  |
  | If they were, we copy the data from the buffer
  | into `a`. The number of copies does not exceed
  | prod(max(a.shape[:-2], b.shape[:-2]) - a.shape[:-2] + 1)
  | and this value is attained by tensors with
  | non-empty batch dimensions.
  |
  | func_t `f` is a callable that is being supplied
  | with Scalar* a_working_ptr, Scalar*
  | b_working_ptr, i64
  | a_linear_batch_idx. a_working_ptr and
  | b_working_ptr can directly be passed to
  | LAPACK/MAGMA routines, and a_linear_batch_idx
  | is an index in the 3d representation which
  | corresponds to the memory a_working_ptr points
  | to, in other words:
  |
  | a_working_ptr == a.view({-1, a.size(-2),
  | a.size(-1)}.select(0,
  | a_linear_batch_idx).data_ptr<Scalar>();
  | a_linear_batch_idx is useful to store metadata
  | related to `a`, such as, for example, its rank
  | or singular values (see linalg_lstsq).
  |
  */
pub fn batch_iterator_with_broadcasting<Scalar, func_t>(
        a: &Tensor,
        b: &Tensor,
        f: &Func)  {

    todo!();
        /*
            IntArrayRef a_batch_sizes(a.sizes().data(), a.dim() - 2);
      IntArrayRef b_batch_sizes(b.sizes().data(), b.dim() - 2);

      auto a_linear_batch_idx = arange(batchCount(a)).view(a_batch_sizes);
      auto b_linear_batch_idx = arange(batchCount(b)).view(b_batch_sizes);

      TensorIterator iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .add_output(b_linear_batch_idx)
        .add_input(a_linear_batch_idx)
        .build();

      auto m = a.size(-2);
      auto n = a.size(-1);
      auto a_3d = a.view({batchCount(a), m, n});
      auto b_3d = b.view({batchCount(b), b.size(-2), b.size(-1)});

      auto a_broadcasts_over_b = (a_batch_sizes != b_batch_sizes);
      Tensor a_buffer, a_was_accessed, a_buffer_3d;
      function<void(i64)> check_if_copy_needed_for_a
        = [](i64 a_curr_linear_batch_idx){};
      if (a_broadcasts_over_b) {
        a_buffer = empty_strided(a.sizes(), a.strides(), a.options())
          .copy_(a);
        a_was_accessed = zeros(batchCount(a), kBool);
        a_buffer_3d = a_buffer.view({batchCount(a), m, n});
        check_if_copy_needed_for_a = [&](i64 a_curr_linear_batch_idx) {
          auto* a_was_accessed_flag = a_was_accessed
            .select(0, a_curr_linear_batch_idx)
            .data_ptr<bool>();
          if (!(*a_was_accessed_flag)) {
            *a_was_accessed_flag = true;
          }
          else {
            a_3d.select(0, a_curr_linear_batch_idx)
              .copy_(a_buffer_3d.select(0, a_curr_linear_batch_idx));
          }
        };
      }

      auto loop = [&](char** data, const i64* strides, i64 nelems) {
        auto* b_batch_idx_ptr = data[0];
        auto* a_batch_idx_ptr = data[1];

        for (i64 elem = 0; elem < nelems; ++elem) {
          auto b_curr_linear_batch_idx = *reinterpret_cast<i64*>(b_batch_idx_ptr);
          auto a_curr_linear_batch_idx = *reinterpret_cast<i64*>(a_batch_idx_ptr);

          check_if_copy_needed_for_a(a_curr_linear_batch_idx);

          auto* a_working_ptr = a_3d.select(0, a_curr_linear_batch_idx)
            .data_ptr<Scalar>();
          auto* b_working_ptr = b_3d.select(0, b_curr_linear_batch_idx)
            .data_ptr<Scalar>();
          f(a_working_ptr, b_working_ptr, a_curr_linear_batch_idx);

          b_batch_idx_ptr += strides[0];
          a_batch_idx_ptr += strides[1];
        }
      };
      iter.serial_for_each(loop, {0, batchCount(b)});
        */
}

/**
  | Returns the epsilon value for floating
  | types except half
  |
  */
#[inline] pub fn get_epsilon(sc_type: &ScalarType) -> f64 {
    
    todo!();
        /*
            switch (sc_type) {
        case ScalarType::Float:
          return static_cast<double>(numeric_limits<float>::epsilon());
        case ScalarType::Double:
          return numeric_limits<double>::epsilon();
        default:
          AT_ERROR("This function doesn't handle types other than float and double");
      }
        */
}

/**
  | Validates input shapes and devices for linear
  | solve methods (solve, cholesky_solve, lu_solve,
  | triangular_solve)
  |
  */
#[inline] pub fn linear_solve_check_inputs(
        self_: &Tensor,
        A:     &Tensor,
        name:  *const u8)  {
    
    todo!();
        /*
            TORCH_CHECK(self.device() == A.device(),
                  "Expected b and A to be on the same device, but found b on ",
                  self.device(), " and A on ", A.device(), " instead.");

      TORCH_CHECK(self.scalar_type() == A.scalar_type(),
                  "Expected b and A to have the same dtype, but found b of type ",
                  self.scalar_type(), " and A of type ", A.scalar_type(), " instead.");

      TORCH_CHECK(A.size(-1) == A.size(-2),
                  "A must be batches of square matrices, "
                  "but they are ", A.size(-1), " by ", A.size(-2), " matrices");

      TORCH_CHECK(A.size(-1) == self.size(-2),
                  "Incompatible matrix sizes for ", name, ": each A "
                  "matrix is ", A.size(-1), " by ", A.size(-1),
                  " but each b matrix is ", self.size(-2), " by ", self.size(-1));
        */
}

/**
  | Validates input shapes for operations on
  | batches of square matrices (inverse, cholesky,
  | symeig, eig)
  |
  */
#[inline] pub fn square_check_inputs(self_: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(self.dim() >= 2, "Tensor of matrices must have at least 2 dimensions. ");
      TORCH_CHECK(self.size(-1) == self.size(-2),
                  "A must be batches of square matrices, "
                  "but they are ", self.size(-1), " by ", self.size(-2), " matrices");
        */
}

/**
  | Given a vector of i64 infos, obtained
  | after a batch operations, this function
  | checks if the computation over all these
  | batches has been successful (info =
  | 0) or not, and report in case of the latter.
  |
  */
#[inline] pub fn batch_check_errors(
        infos:          &mut Vec<i64>,
        name:           *const u8,
        allow_singular: bool)  {
    let allow_singular: bool = allow_singular.unwrap_or(false);

    todo!();
        /*
            for (usize i = 0; i < infos.size(); i++) {
        auto info = infos[i];
        if (info < 0) {
          AT_ERROR(name, ": For batch ", i, ": Argument ", -info, " has illegal value");
        } else if (info > 0) {
          if (strstr(name, "svd")) {
            AT_ERROR(name, ": the updating process of SBDSDC did not converge (error: ", info, ")");
          } else if (strstr(name, "symeig") || strstr(name, "syevd")) {
            AT_ERROR(name, ": For batch ", i, ": the algorithm failed to converge; ", info,
                     " off-diagonal elements of an intermediate tridiagonal form did not converge to zero.");
          } else if (!allow_singular) {
            AT_ERROR(name, ": For batch ", i, ": U(", info, ",", info, ") is zero, singular U.");
          }
        }
      }
        */
}

/**
  | This is an overloaded case of the previous
  | function for a tensor of infos.
  |
  */
#[inline] pub fn batch_check_errors_with_tensor_of_infos(
        infos:          &Tensor,
        name:           *const u8,
        allow_singular: bool,
        info_per_batch: i32)  {

    let allow_singular: bool = allow_singular.unwrap_or(false);
    let info_per_batch: i32 = info_per_batch.unwrap_or(1);

    todo!();
        /*
            auto batch_size = infos.numel();
      auto infos_cpu = infos.to(kCPU);
      auto infos_data = infos_cpu.data_ptr<int>();
      for (i64 i = 0; i < batch_size; i++) {
        auto info = infos_data[i];
        if (info < 0) {
          AT_ERROR(name, ": For batch ", i/info_per_batch, ": Argument ", -info, " has illegal value");
        } else if (!allow_singular && info > 0) {
          AT_ERROR(name, ": For batch ", i/info_per_batch, ": U(", info, ",", info, ") is zero, singular U.");
        }
      }
        */
}

/**
  | Given a info int, obtained after a single
  | operation, this function check if the
  | computation has been successful (info
  | = 0) or not, and report in case of the latter.
  |
  */
#[inline] pub fn single_check_errors(
        info:           i64,
        name:           *const u8,
        allow_singular: bool)  {
    let allow_singular: bool = allow_singular.unwrap_or(false);

    todo!();
        /*
            if (info < 0) {
        AT_ERROR(name, ": Argument ", -info, " has illegal value");
      } else if (info > 0) {
        if (strstr(name, "svd")) {
          AT_ERROR(name, ": the updating process of SBDSDC did not converge (error: ", info, ")");
        } else if (strstr(name, "eig")) { // this catches both "eig" and "symeig"
          AT_ERROR(name, ": the algorithm failed to converge; ", info,
                   " off-diagonal elements of an intermediate tridiagonal form did not converge to zero.");
        } else if (!allow_singular) {
          AT_ERROR(name, ": U(", info, ",", info, ") is zero, singular U.");
        }
      }
        */
}

/**
  | Checks if all the Tensors in a &[Tensor]
  | are of the same dimensions
  |
  */
#[inline] pub fn check_all_same_dim(
        tensors: &[Tensor],
        dim:     i64)  {
    
    todo!();
        /*
            for (auto &t : tensors) {
        TORCH_CHECK(t.dim() == dim, "Tensor dimension is ", t.dim(), ", expected ", dim, " instead.");
      }
        */
}

#[inline] pub fn linalg_broadcast_batch_dims(
        arg1: &Tensor,
        arg2: &Tensor,
        name: *const u8) -> (Tensor,Tensor) {
    
    todo!();
        /*
            linearSolveCheckInputs(arg1, arg2, name);

      // broadcast the batch dimensions of arg1 and arg2.
      IntArrayRef arg1_batch_sizes(arg1.sizes().data(), arg1.ndimension() - 2);
      IntArrayRef arg2_batch_sizes(arg2.sizes().data(), arg2.ndimension() - 2);
      vector<i64> expand_batch_portion = infer_size(arg1_batch_sizes, arg2_batch_sizes);

      vector<i64> arg1_expand_size({expand_batch_portion});
      arg1_expand_size.insert(arg1_expand_size.end(), { arg1.size(-2), arg1.size(-1) });

      vector<i64> arg2_expand_size({expand_batch_portion});
      arg2_expand_size.insert(arg2_expand_size.end(), { arg2.size(-2), arg2.size(-1) });

      Tensor arg1_broadcasted  = arg1.expand(arg1_expand_size);
      Tensor arg2_broadcasted = arg2.expand(arg2_expand_size);
      return make_tuple(arg1_broadcasted, arg2_broadcasted);
        */
}

#[inline] pub fn broadcast_batch_size(
        t1:           &Tensor,
        t2:           &Tensor,
        n_batch_dims: i64) -> Vec<i64> {
    
    todo!();
        /*
            IntArrayRef t1_batch_sizes(t1.sizes().data(), n_batch_dims);
      IntArrayRef t2_batch_sizes(t2.sizes().data(), n_batch_dims);
      auto broadcasted_batch_sizes = infer_size(t1_batch_sizes, t2_batch_sizes);
      return broadcasted_batch_sizes;
        */
}

/**
  | Return a permutation with the given
  | axes moved to the end.
  |
  */
#[inline] pub fn move_to_end(
        self_: &Tensor,
        axes:  &[i32]) -> Tensor {
    
    todo!();
        /*
            const vector<i64> a = axes.vec();
      const i64 ndim = self.ndimension();
      vector<i64> perm;

      for (i64 i = 0; i < ndim; i++) {
        auto it = find(a.begin(), a.end(), i);
        if (it == a.end()) {
           perm.push_back(i);
        }
      }
      for (auto i : a) {
        perm.push_back(i);
      }

      TORCH_CHECK((i64)perm.size() == ndim,
        "duplicate or invalid axis in 'dim' argument for tensor with ndim==", ndim);

      return self.permute(perm);
        */
}

/**
  | parse the "mode" param in linalg_qr:
  | return a tuple of bools (compute_q,
  | reduced)
  |
  */
#[inline] pub fn parse_qr_mode(mode: StringView) -> (bool,bool) {
    
    todo!();
        /*
            bool compute_q;
      bool reduced;
      if (mode == "reduced") {
        compute_q = true;
        reduced = true;
      } else if (mode == "complete") {
        compute_q = true;
        reduced = false;
      } else if (mode == "r") {
        compute_q = false;
        reduced = true; // this is actually irrelevant in this mode
      } else {
          TORCH_CHECK(false, "qr received unrecognized mode '", mode,
                      "' but expected one of 'reduced' (default), 'r', or 'complete'");
      }
      return make_tuple(compute_q, reduced);
        */
}

/**
  | Function to compute sizes, strides
  | and the extra columns for the Q matrix
  | in the QR Decomposition
  |
  */
#[inline] pub fn compute_geometry_for_q(
        input:   &Tensor,
        reduced: bool) -> (Vec<i64>,Vec<i64>,i64) {
    
    todo!();
        /*
            i64 m = input.size(-2), n = input.size(-1);
      i64 n_columns_q;

      // We need to compute the required size of Q based on the `reduced` option
      auto q_sizes = input.sizes().vec();
      if (!reduced && m > n) {
        q_sizes[input.dim() - 1] = m;
        n_columns_q = m;
      } else {
        q_sizes[input.dim() - 1] = n;
        n_columns_q = min(m, n);
      }
      auto q_strides = defaultStrides(q_sizes);

      // Q should be a column-major or a batch of column-major matrices
      // ... x m x n will have strides: ...., n, 1
      // We require: ...., 1, m
      q_strides[input.dim() - 1] = m;
      q_strides[input.dim() - 2] = 1;
      return make_tuple(q_sizes, q_strides, n_columns_q);
        */
}

/**
  | Function to generate empty tensors
  | of required size, strides and dtype
  | for the SVD operation
  |
  */
#[inline] pub fn create_u_s_vt(
        input:            &Tensor,
        some:             bool,
        compute_uv:       bool,
        svd_use_cusolver: bool) -> (Tensor,Tensor,Tensor) {
    let svd_use_cusolver: bool = svd_use_cusolver.unwrap_or(false);

    todo!();
        /*
            // U, S, VT are initialized as empty tensors.
      // For CPU LAPACK and GPU MAGMA backend, the tensors are initialized on CPU.
      // For GPU cuSOLVER backend, the tensors are initialized on GPU.
      const auto usvt_device = svd_use_cusolver ? kCUDA : kCPU;

      auto sizes = input.sizes().vec();
      i64 m = input.size(-2), n = input.size(-1);

      sizes[input.dim() - 1] = (compute_uv && some) ? min(m, n) : m;
      auto strides = defaultStrides(sizes);
      // U should be a column-major or a batch of column-major matrices
      // ... x m x ucol will have strides: ...., ucol, 1
      // We require: ...., 1, m
      strides[input.dim() - 1] = m;
      strides[input.dim() - 2] = 1;

      Tensor U_empty = empty_strided(sizes, strides, input.options().device(usvt_device));
      U_empty.zero_();

      // VT should be a column-major or a batch of column-major matrices
      sizes[input.dim() - 2] = n;
      sizes[input.dim() - 1] = n;
      // VT should be a column-major or a batch of column-major matrices
      Tensor VT_empty = zeros(sizes, input.options().device(usvt_device));
      VT_empty.transpose_(-2, -1);

      sizes.pop_back();
      sizes[input.dim() - 2] = min(m, n);
      ScalarType dtype = toValueType(typeMetaToScalarType(input.dtype()));
      Tensor S_empty = empty(sizes, input.options().dtype(dtype).device(usvt_device));

      return tuple<Tensor, Tensor, Tensor>(U_empty, S_empty, VT_empty);
        */
}

/**
  | Function used instead of .to so that the
  | original strides are retained .to doesn't
  | retain strides and make the output tensor
  | contiguous
  |
  */
#[inline] pub fn same_stride_to(
        original_tensor: &Tensor,
        options:         &TensorOptions) -> Tensor {
    
    todo!();
        /*
            auto strided_to = empty_strided(original_tensor.sizes(),
                                          original_tensor.strides(),
                                          options);
      strided_to.copy_(original_tensor);
      return strided_to;
        */
}

/**
  | Creates a dimension permutation array that can
  | be given to `permute()`, which will shift the
  | two specified dimensions to the end of
  | a tensor, without changing the order of the
  | other dimensions. `dim1` will be placed at the
  | very end, and `dim0` will be placed just to the
  | left of it.
  |
  | For instance, given a 4-D tensor, dimensions
  | 1 and 3 can be shifted to the end by calling
  | `create_dim_backshift_permutation(1, 3,
  | 4)`. The resulting vector will be `vec(0, 2, 1, 3)`.
  */
#[inline] pub fn create_dim_backshift_permutation(
        dim0: i64,
        dim1: i64,
        ndim: i64) -> Vec<i64> {
    
    todo!();
        /*
            TORCH_CHECK(
        (dim0 != dim1) && (dim0 < ndim) && (dim0 >= 0) && (dim1 < ndim) && (dim1 >= 0),
        "duplicate or invalid dimensions");
      vector<i64> permutation(ndim);
      i64 cur_permuted_dim = 0;
      for (i64 dim_ind = 0; dim_ind < ndim; dim_ind++) {
        if ((dim_ind != dim0) && (dim_ind != dim1)) {
          permutation[cur_permuted_dim++] = dim_ind;
        }
      }
      permutation[cur_permuted_dim++] = dim0;
      permutation[cur_permuted_dim] = dim1;
      return permutation;
        */
}

/**
  | Creates a dimension permutation array that can
  | be given to `permute()`, which will reverse
  | a given permutation.
  |
  | The reverse permutation array is created by
  | swapping the indices and their associated
  | values from the given permutation array.
  |
  */
#[inline] pub fn create_reverse_permutation(permutation: Vec<i64>) -> Vec<i64> {
    
    todo!();
        /*
            i64 ndim = permutation.size();
      vector<i64> reverse_permutation(ndim);
      for (i64 dim_ind = 0; dim_ind < ndim; dim_ind++) {
        reverse_permutation[permutation[dim_ind]] = dim_ind;
      }
      return reverse_permutation;
        */
}

/**
  | Compute R-work array size for MAGMA/LAPACK
  | cgesdd/zgesdd
  |
  | See https://github.com/Reference-LAPACK/lapack/blob/122506cd8b6ce050a200920c3d4c0b153b150fd8/SRC/cgesdd.f#L186
  |
  */
#[inline] pub fn compute_lrw_ork_dim(
        jobz: u8,
        m:    i64,
        n:    i64) -> i64 {
    
    todo!();
        /*
            auto mn = min(m, n);
      auto mx = max(m, n);
      if (jobz == 'N') {
    #ifdef __APPLE__
        // According to `vecLib.framework/Headers/clapack.h` Accelerate.framework is based on LAPACK 3.2.1
        return 7 * mn;
    #else
        // These setting is valid for on LAPACK 3.6+
        return 5 * mn;
    #endif
      }
      if (mx > 10 * mn) {
        return 5 * mn * mn + 5 * mn;
      }
      return max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn);
        */
}

/**
  | This function checks whether the uplo argument
  | input is valid
  |
  | Allowed strings are "u", "U", "l", "L"
  |
  */
#[inline] pub fn check_uplo(uplo: StringView)  {
    
    todo!();
        /*
            // To use toupper safely with plain chars (or signed chars), the argument should first be converted to unsigned char
      char uplo_uppercase = static_cast<char>(toupper(static_cast<unsigned char>(uplo[0])));
      TORCH_CHECK(uplo.size() == 1 && (uplo_uppercase == 'U' || uplo_uppercase == 'L'),
        "Expected UPLO argument to be 'L' or 'U', but got ", uplo);
        */
}

#[inline] pub fn check_same_device(
        fn_name:     &String,
        result:      Tensor,
        input:       Tensor,
        result_name: &String)  {
    let result_name: &String = result_name.unwrap_or("result");

    todo!();
        /*
            TORCH_CHECK(
          result.device() == input.device(),
          fn_name,
          ": Expected ", result_name, " and input tensors to be on the same device, but got ",
          result_name, " on ", result.device(), " and input on ", input.device());
        */
}

/**
  | Check the dtype of result and input tensors
  | (for _out variants).
  |
  | Most linear algebra functions have the same
  | dtype for input and output (either floating or
  | complex type input), so we can check whether
  | input's dtype can be casted to result's dtype.
  |
  | According to
  | https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
  | canCast is used for checking the "safe copy"
  | dtype requirements.
  |
  */
#[inline] pub fn check_linalg_compatible_dtype_tensor(
    fn_name:     &String,
    result:      Tensor,
    input:       Tensor,
    result_name: &String) 
{
    let result_name: &String = result_name.unwrap_or("result");

    todo!();
        /*
            bool can_cast = canCast(input.scalar_type(), result.scalar_type());
      TORCH_CHECK(
          can_cast,
          fn_name,
          ": Expected ", result_name, " to be safely castable from ", input.scalar_type(), " dtype, but got ",
          result_name, " with dtype ", result.scalar_type());
        */
}

/**
  | Alternatively, we can check whether the
  | specific expected output type (result_type) can
  | be safely casted to out tensor dtype (out_type)
  |
  */
#[inline] pub fn check_linalg_compatible_dtype(
        fn_name:     &String,
        out_type:    ScalarType,
        result_type: ScalarType,
        out_name:    &String)  {
    let out_name: &String = out_name.unwrap_or("result");

    todo!();
        /*
            bool can_cast = canCast(result_type, out_type);
      TORCH_CHECK(
          can_cast,
          fn_name,
          ": Expected ", out_name, " to be safely castable from ", result_type, " dtype, but got ",
          out_name, " with dtype ", out_type);
        */
}

/**
  | Two types of 'other' tensors are supported
  | when solving a system of linear equations
  | matmul(input, x) = other: 1-dimensional
  | (1D) tensor or batch of 1D tensors (vector
  | case) 2-dimensional (2D) tensor or
  | batch of 2D tensors (matrix case).
  | 
  | The original torch.solve supported
  | only the matrix case, while NumPy works
  | for both cases.
  | 
  | For the batched input we need to be able
  | to distinguish them.
  | 
  | Let input.shape = (batch_dimensions,
  | m, n), then 'other' is of vector type
  | if other.shape == (batch_dimensions,
  | m).
  | 
  | This rule is compatible with NumPy,
  | see https://github.com/numpy/numpy/blob/v1.20.0/numpy/linalg/linalg.py#L384-L389
  |
  */
#[inline] pub fn linalg_solve_is_vector_rhs(
        input: &Tensor,
        other: &Tensor) -> bool {
    
    todo!();
        /*
            auto expected_batched_rhs_shape = IntArrayRef(input.sizes().data(), input.dim() - 1); // input.shape[:-1]
      bool vector_case = other.dim() == 1 || (input.dim() - 1 == other.dim() && other.sizes().equals(expected_batched_rhs_shape));
      return vector_case;
        */
}
