crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TriangularOpsUtils.h]

/**
  | Given batches of matrices with arbitrary
  | batch dim, computes the number of batches
  | for Triu and Tril. This ignores stride
  | 0 dimension
  |
  */
#[inline] pub fn batch_count_tril_triu(batched_matrices: &Tensor) -> i64 {
    
    todo!();
        /*
            i64 result = 1;
      for (i64 i = 0; i < batched_matrices.ndimension() - 2; i++) {
        if (batched_matrices.stride(i) != 0) {
          result *= batched_matrices.size(i);
        }
      }
      return result;
        */
}

/**
  | Checks a necessary property for the
  | triu and tril implementations, hence
  | the name.
  | 
  | Here batch contiguity is checked for
  | tensors with greater than 4 dimensions.
  | 
  | Contiguous tensors and tensors with
  | less than 3 dimensions pass this check
  |
  */
#[inline] pub fn check_tril_triu_batch_contiguous(
        tensor:            &Tensor,
        allow_zero_stride: bool) -> (bool,Tensor) {
    
    todo!();
        /*
            // Complete contiguity is the most desired property, which is why
      // we return true if the tensor is contiguous
      if (tensor.is_contiguous()) {
        auto default_strides_for_size = defaultStrides(tensor.sizes());
        if (tensor.strides() == default_strides_for_size) {
          return make_tuple(true, tensor);
        } else {
          return make_tuple(false, tensor.as_strided(tensor.sizes(), default_strides_for_size));
        }
      }

      i64 dims = tensor.dim();

      // Tensors with dimension less than 4 are handled by default
      if (allow_zero_stride && dims <= 3) {
        return make_tuple(true, tensor);
      }

      i64 expected_stride = tensor.size(-1) * tensor.size(-2);
      for (i64 i = dims - 3; i >= 0; i--) {
        // Skip trivial dimension;
        if (allow_zero_stride && i == 0 && (tensor.stride(i) == 0 || tensor.size(i) == 1)) {
          continue;
        }
        if (expected_stride != tensor.stride(i)) {
          return make_tuple(false, tensor.contiguous());
        }
        expected_stride *= tensor.size(i);
      }
      return make_tuple(true, tensor);
        */
}
