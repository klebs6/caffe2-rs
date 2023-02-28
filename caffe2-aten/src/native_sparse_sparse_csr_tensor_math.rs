crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/sparse/SparseCsrTensorMath.cpp]

/**
  | certain utiliy functions are usable
  | from sparse COO.
  |
  */
pub fn is_mkl_supported() -> bool {
    
    todo!();
        /*
            #ifdef _MSC_VER
      return false;
    #elif  __APPLE__ || __MACH__
      return false;
    #else
      return true;
    #endif
        */
}

/**
  | Only accept squares sparse matrices or dense
  | input as a vector
  |
  | TODO: Check what happens with MKL, the output
  | error reported with non square matrices tends
  | to be high
  |
  | See:
  | https://github.com/pytorch/pytorch/issues/58770
  */
pub fn is_square_or_vec(
        dim_i: i64,
        dim_j: i64,
        dim_k: i64) -> bool {
    
    todo!();
        /*
            return (dim_i == dim_k  && dim_k == dim_j) || (dim_i == dim_j && dim_k == 1);
        */
}

pub fn addmm_out_sparse_dense_worker<Scalar>(
        nnz:         i64,
        dim_i:       i64,
        dim_j:       i64,
        dim_k:       i64,
        r:           &mut Tensor,
        beta:        Scalar,
        t:           &Tensor,
        alpha:       Scalar,
        csr:         &Tensor,
        col_indices: &Tensor,
        values:      &Tensor,
        dense:       &Tensor)  {

    todo!();
        /*
            Scalar cast_alpha = alpha.to<Scalar>();
      Scalar cast_beta = beta.to<Scalar>();
      if (cast_beta == 0) {
        r.zero_();
      } else if (cast_beta == 1) {
        if (!is_same_tensor(r, t)) {
          r.copy_(t);
        }
      } else {
        mul_out(r, t, scalar_to_tensor(beta));
      }
      AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "csr_mm_crow_indices", [&]() {
        auto csr_accessor = csr.accessor<Index, 1>();
        auto col_indices_accessor = col_indices.accessor<Index, 1>();

        auto values_accessor = values.accessor<Scalar, 1>();
        Scalar* dense_ptr = dense.data_ptr<Scalar>();
        Scalar* r_ptr = r.data_ptr<Scalar>();

        i64 dense_stride0 = dense.stride(0);
        i64 dense_stride1 = dense.stride(1);
        i64 r_stride0 = r.stride(0);
        i64 r_stride1 = r.stride(1);

        parallel_for(
            0,
            dim_i,
            internal::GRAIN_SIZE,
            [&](i64 irow_start, i64 irow_end) {
                for (Index h = irow_start; h < irow_end; ++h) {
                  Index i_start = csr_accessor[h];
                  Index i_end = csr_accessor[h+1];
                  for (Index i = i_start; i < i_end; i++) {
                    Scalar val = values_accessor[i];
                    Index col = col_indices_accessor[i];
                    native::cpublas::axpy<Scalar>(dim_k,
                        cast_alpha * val,
                        dense_ptr + col * dense_stride0, dense_stride1,
                        r_ptr + h * r_stride0, r_stride1);
                  }
                }
        });
      });
        */
}

// Functions for matrix multiplication.
pub fn addmm_out_sparse_csr_dense_cpu(
        self_:  &Tensor,
        sparse: &SparseCsrTensor,
        dense:  &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar,
        r:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(sparse.is_sparse_csr());
      Tensor t = *expand_size(self, {sparse.size(0), dense.size(1)}, "addmm_out_sparse_csr");

      TORCH_CHECK(!t.is_cuda(),  "Expected all tensors to be on the same device. addmm expected 't' to be CPU tensor, but got CUDA tensor");
      TORCH_CHECK(
          !r.is_cuda(),
          "Expected all tensors to be on the same device. addmm: expected 'out' to be CPU tensor, but got CUDA tensor");
      TORCH_CHECK(
          !sparse.is_cuda(),
          "Expected all tensors to be on the same device. addmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
      TORCH_CHECK(
          !dense.is_cuda(),
          "Expected all tensors to be on the same device. addmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

      TORCH_CHECK(
          sparse.dim() == 2,
          "addmm: 2-D matrices expected, got ",
          sparse.dim(),
          "D tensor");
      TORCH_CHECK(
          dense.dim() == 2,
          "addmm: 2-D matrices expected, got ",
          dense.dim(),
          "D tensor");

      TORCH_CHECK(
          r.is_contiguous(),
          "out argument must be contiguous, but got: ",
          r.suggest_memory_format());

      // ixj * jxk = ixk
      i64 dim_i = sparse.size(0);
      i64 dim_j = sparse.size(1);
      i64 dim_k = dense.size(1);

      TORCH_CHECK(
          dense.size(0) == dim_j,
          "addmm: Expected dense matrix (op2) size(0)=",
          dim_j,
          ", got ",
          dense.size(0));

      resize_output(r, {dim_i, dim_k});
      auto col_indices = sparse.col_indices();
      auto crow_indices = sparse.crow_indices();
      auto values = sparse.values();
      i64 nnz        = sparse._nnz();
      if (nnz == 0) {
        mul_out(r, t, scalar_tensor(beta, r.options()));
        return r;
      }
      // Do not use MKL for Windows due to linking issues with sparse MKL routines.
      if (hasMKL() && is_mkl_supported() && is_square_or_vec(dim_i, dim_j, dim_k)) {
        AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "addmm_sparse_dense", [&] {
            Scalar cast_beta = beta.to<Scalar>();
            if (cast_beta == 0) {
              r.zero_();
            } else if (cast_beta == 1) {
              if (!is_same_tensor(r, t)) {
                r.copy_(t);
              }
            } else {
              mul_out(r, t, scalar_to_tensor(beta));
            }
            // r = r + alpha * sparse * dense
            _sparse_mm_mkl_(r, sparse, dense, t, alpha, Scalar(static_cast<Scalar>(1.0)));
        });
      } else {
        // r = beta * t + alpha * sparse * dense
        AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "addmm_sparse_dense", [&] {
            s_addmm_out_sparse_dense_worker<Scalar>(nnz, dim_i, dim_j, dim_k, r, beta, t, alpha, crow_indices, col_indices, values, dense);
        });
      }
      return r;
        */
}

pub fn addmm_sparse_csr_dense(
        self_:  &Tensor,
        sparse: &SparseCsrTensor,
        dense:  &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor r = empty({0}, self.options());
      addmm_out(r, self, sparse, dense, beta, alpha);
      return r;
        */
}

pub fn sparse_csr_mm_out(
        sparse: &SparseCsrTensor,
        dense:  &Tensor,
        result: &mut SparseCsrTensor) -> &mut SparseCsrTensor {
    
    todo!();
        /*
            Tensor t = zeros({}, dense.options());
      return addmm_out(result, t, sparse, dense, 0.0, 1.0); // redispatch!
        */
}

pub fn sparse_csr_addmm(
        t:      &Tensor,
        sparse: &SparseCsrTensor,
        dense:  &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> Tensor {
    
    todo!();
        /*
            // _sparse_addmm forward is functionally equivalent to addmm; it's
      // just the backward that is different.  This technically does an
      // unnecessary redispatch, I was too lazy to make it not do that
      return addmm(t, sparse, dense, beta, alpha);
        */
}

// Functions for element-wise addition.
pub fn add_sparse_csr_mut(
        self_: &Tensor,
        other: &Tensor,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            auto commonDtype = result_type(self, other);
      alpha_check(commonDtype, alpha);
      Tensor result = empty({0}, self.options().dtype(commonDtype));
      return add_out(result, self, other, alpha); // redispatch!
        */
}

pub fn add_sparse_csr(
        self_: &mut Tensor,
        other: &Tensor,
        alpha: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return add_out(self, self, other, alpha); // redispatch!
        */
}

pub fn add_out_dense_sparse_csr_cpu(
        out:   &mut Tensor,
        dense: &Tensor,
        src:   &SparseCsrTensor,
        alpha: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(dense.layout() == kStrided);
      TORCH_INTERNAL_ASSERT(src.is_sparse_csr());
      TORCH_INTERNAL_ASSERT(dense.device() == kCPU);

      TORCH_CHECK(
          out.is_contiguous(),
          "out argument must be contiguous, but got: ",
          out.suggest_memory_format());
      TORCH_CHECK(
          out.device() == kCPU,
          "add: expected 'out' to be CPU tensor, but got tensor on device: ",
          out.device());
      TORCH_CHECK(
          src.device() == kCPU,
          "add: expected 'other' to be a CPU tensor, but got tensor on device: ",
          src.device());

      TORCH_CHECK(
          dense.sizes().equals(src.sizes()),
          "add: expected 'self' and 'other' to have same size, but self has size ",
          dense.sizes(),
          " while other has size ",
          src.sizes(),
          " (FYI: op2-sparse addition does not currently support broadcasting)");

      auto commonDtype = promoteTypes(dense.scalar_type(), src.scalar_type());
      TORCH_CHECK(
          canCast(commonDtype, out.scalar_type()),
          "Can't convert result type ",
          commonDtype,
          " to output ",
          out.scalar_type(),
          " in add operation");

      auto src_values = src.values();
      auto src_crow_indices = src.crow_indices();
      auto src_col_indices = src.col_indices();

      resize_output(out, dense.sizes());

      Tensor resultBuffer = out;
      Tensor valuesBuffer = src_values.to(commonDtype);

      if (out.scalar_type() != commonDtype) {
        resultBuffer = dense.to(commonDtype);
      } else if (!is_same_tensor(out, dense)) {
        resultBuffer.copy_(dense);
      }

      AT_DISPATCH_ALL_TYPES(
          commonDtype,
          "add_out_op2_sparse_csr",
          [&valuesBuffer, &resultBuffer, &alpha, &src_crow_indices, &src_col_indices]() {
            AT_DISPATCH_INDEX_TYPES(
                src_crow_indices.scalar_type(),
                "csr_add_out_crow_indices",
                [&valuesBuffer, &resultBuffer, &alpha, &src_crow_indices, &src_col_indices]() {
                  auto values_accessor = valuesBuffer.accessor<Scalar, 1>();
                  Scalar* out_ptr = resultBuffer.data_ptr<Scalar>();
                  Scalar cast_value = alpha.to<Scalar>();

                  auto crow_indices_accessor =
                      src_crow_indices.accessor<Index, 1>();
                  auto col_indices_accessor =
                      src_col_indices.accessor<Index, 1>();
                  auto out_strides0 = resultBuffer.strides()[0];
                  auto out_strides1 = resultBuffer.strides()[1];

                  for (Index irow = 0; irow < src_crow_indices.size(0) - 1;
                       ++irow) {
                    Index start_index = crow_indices_accessor[irow];
                    Index end_index = crow_indices_accessor[irow + 1];

                    for (Index i = start_index; i < end_index; ++i) {
                      auto icol = col_indices_accessor[i];
                      auto index = resultBuffer.storage_offset() + irow * out_strides0 +
                          icol * out_strides1;
                      out_ptr[index] += cast_value * values_accessor[i];
                    }
                  }
                });
          });
      if (out.scalar_type() != commonDtype) {
        out.copy_(resultBuffer);
      }
      return out;
        */
}

pub fn add_out_sparse_csr_cpu(
        self_: &Tensor,
        other: &SparseCsrTensor,
        alpha: &Scalar,
        out:   &mut SparseCsrTensor) -> &mut Tensor {
    
    todo!();
        /*
            if (self.layout() == kStrided) {
        return add_out_dense_sparse_csr_cpu(out, self, other, alpha);
      } else {
        TORCH_CHECK(
            false,
            "NotImplementedError: Addition of sparse CSR tensors is not yet implemented.")
      }
      return out;
        */
}

