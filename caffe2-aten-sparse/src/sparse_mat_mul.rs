crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/sparse/SparseMatMul.cpp]

/**
  | This is an implementation of the SMMP
  | algorithm: "Sparse Matrix Multiplication
  | Package (SMMP)"
  | 
  | Randolph E. Bank and Craig C. Douglas
  | https://doi.org/10.1007/BF02070824
  |
  */
pub fn csr_to_coo(
        n_row: i64,
        ap:    &[i64],
        bi:    &[i64])  {

    
    todo!();
        /*
            /*
        Expands a compressed row pointer into a row indices array
        Inputs:
          `n_row` is the number of rows in `Ap`
          `Ap` is the row pointer

        Output:
          `Bi` is the row indices
      */
      for (i64 i = 0; i < n_row; i++) {
        for (i64 jj = Ap[i]; jj < Ap[i + 1]; jj++) {
          Bi[jj] = i;
        }
      }
        */
}

pub fn csr_matmult_maxnnz(
    n_row: i64,
    n_col: i64,
    ap:    &[i64],
    aj:    &[i64],
    bp:    &[i64],
    bj:    &[i64]) -> i64 {


    todo!();
        /*
            /*
        Compute needed buffer size for matrix `C` in `C = A@B` operation.

        The matrices should be in proper CSR structure, and their dimensions
        should be compatible.
      */
      vector<i64> mask(n_col, -1);
      i64 nnz = 0;
      for (i64 i = 0; i < n_row; i++) {
        i64 row_nnz = 0;

        for (i64 jj = Ap[i]; jj < Ap[i + 1]; jj++) {
          i64 j = Aj[jj];
          for (i64 kk = Bp[j]; kk < Bp[j + 1]; kk++) {
            i64 k = Bj[kk];
            if (mask[k] != i) {
              mask[k] = i;
              row_nnz++;
            }
          }
        }
        i64 next_nnz = nnz + row_nnz;
        nnz = next_nnz;
      }
      return nnz;
        */
}

pub fn csr_matmult<Scalar>(
    n_row: i64,
    n_col: i64,
    ap:    &[i64],
    aj:    &[i64],
    ax:    &[Scalar],
    bp:    &[i64],
    bj:    &[i64],
    bx:    &[Scalar],
    cp:    &[i64],
    cj:    &[i64],
    cx:    &[Scalar])  {

    todo!();
        /*
            /*
        Compute CSR entries for matrix C = A@B.

        The matrices `A` and 'B' should be in proper CSR structure, and their dimensions
        should be compatible.

        Inputs:
          `n_row`         - number of row in A
          `n_col`         - number of columns in B
          `Ap[n_row+1]`   - row pointer
          `Aj[nnz(A)]`    - column indices
          `Ax[nnz(A)]     - nonzeros
          `Bp[?]`         - row pointer
          `Bj[nnz(B)]`    - column indices
          `Bx[nnz(B)]`    - nonzeros
        Outputs:
          `Cp[n_row+1]` - row pointer
          `Cj[nnz(C)]`  - column indices
          `Cx[nnz(C)]`  - nonzeros

        Note:
          Output arrays Cp, Cj, and Cx must be preallocated
      */
      vector<i64> next(n_col, -1);
      vector<Scalar> sums(n_col, 0);

      i64 nnz = 0;

      Cp[0] = 0;

      for (i64 i = 0; i < n_row; i++) {
        i64 head = -2;
        i64 length = 0;

        i64 jj_start = Ap[i];
        i64 jj_end = Ap[i + 1];
        for (i64 jj = jj_start; jj < jj_end; jj++) {
          i64 j = Aj[jj];
          Scalar v = Ax[jj];

          i64 kk_start = Bp[j];
          i64 kk_end = Bp[j + 1];
          for (i64 kk = kk_start; kk < kk_end; kk++) {
            i64 k = Bj[kk];

            sums[k] += v * Bx[kk];

            if (next[k] == -1) {
              next[k] = head;
              head = k;
              length++;
            }
          }
        }

        for (i64 jj = 0; jj < length; jj++) {
          Cj[nnz] = head;
          Cx[nnz] = sums[head];
          nnz++;

          i64 temp = head;
          head = next[head];

          next[temp] = -1; // clear arrays
          sums[temp] = 0;
        }

        Cp[i + 1] = nnz;
      }
        */
}

pub fn sparse_matmul_kernel<Scalar>(
    output: &mut Tensor,
    mat1:   &Tensor,
    mat2:   &Tensor)  {

    todo!();
        /*
            /*
        Computes  the sparse-sparse matrix multiplication between `mat1` and `mat2`, which are sparse tensors in COO format.
      */

      auto M = mat1.size(0);
      auto K = mat1.size(1);
      auto N = mat2.size(1);

      auto mat1_indices_ = mat1._indices().contiguous();
      auto mat1_values = mat1._values().contiguous();
      Tensor mat1_row_indices = mat1_indices_.select(0, 0);
      Tensor mat1_col_indices = mat1_indices_.select(0, 1);

      Tensor mat1_indptr = coo_to_csr(mat1_row_indices.data_ptr<i64>(), M, mat1._nnz());

      auto mat2_indices_ = mat2._indices().contiguous();
      auto mat2_values = mat2._values().contiguous();
      Tensor mat2_row_indices = mat2_indices_.select(0, 0);
      Tensor mat2_col_indices = mat2_indices_.select(0, 1);

      Tensor mat2_indptr = coo_to_csr(mat2_row_indices.data_ptr<i64>(), K, mat2._nnz());

      auto nnz = _csr_matmult_maxnnz(M, N, mat1_indptr.data_ptr<i64>(), mat1_col_indices.data_ptr<i64>(),
          mat2_indptr.data_ptr<i64>(), mat2_col_indices.data_ptr<i64>());

      auto output_indices = output._indices();
      auto output_values = output._values();

      Tensor output_indptr = empty({M + 1}, kLong);
      native::resize_output(output_indices, {2, nnz});
      native::resize_output(output_values, nnz);

      Tensor output_row_indices = output_indices.select(0, 0);
      Tensor output_col_indices = output_indices.select(0, 1);

      _csr_matmult(M, N, mat1_indptr.data_ptr<i64>(), mat1_col_indices.data_ptr<i64>(), mat1_values.data_ptr<Scalar>(),
      mat2_indptr.data_ptr<i64>(), mat2_col_indices.data_ptr<i64>(), mat2_values.data_ptr<Scalar>(),
      output_indptr.data_ptr<i64>(), output_col_indices.data_ptr<i64>(), output_values.data_ptr<Scalar>());

      csr_to_coo(M, output_indptr.data_ptr<i64>(), output_row_indices.data_ptr<i64>());
        */
}

pub fn sparse_sparse_matmul_cpu(
    mat1: &Tensor,
    mat2: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
      TORCH_INTERNAL_ASSERT(mat2_.is_sparse());
      TORCH_CHECK(mat1_.dim() == 2);
      TORCH_CHECK(mat2_.dim() == 2);
      TORCH_CHECK(mat1_.dense_dim() == 0, "sparse_sparse_matmul_cpu: scalar values expected, got ", mat1_.dense_dim(), "D values");
      TORCH_CHECK(mat2_.dense_dim() == 0, "sparse_sparse_matmul_cpu: scalar values expected, got ", mat2_.dense_dim(), "D values");

      TORCH_CHECK(
          mat1_.size(1) == mat2_.size(0), "mat1 and mat2 shapes cannot be multiplied (",
          mat1_.size(0), "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1), ")");

      TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(),
               "mat1 dtype ", mat1_.scalar_type(), " does not match mat2 dtype ", mat2_.scalar_type());

      auto output = native::empty_like(mat1_);
      output.sparse_resize_and_clear_({mat1_.size(0), mat2_.size(1)}, mat1_.sparse_dim(), 0);

      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(mat1_.scalar_type(), "sparse_matmul", [&] {
        sparse_matmul_kernel<Scalar>(output, mat1_.coalesce(), mat2_.coalesce());
      });
      return output;
        */
}
