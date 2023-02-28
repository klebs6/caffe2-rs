crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/WrapDimUtils.h]

#[inline] pub fn maybe_wrap_dim_with_wrap_scalar(
        dim:           i64,
        dim_post_expr: i64,
        wrap_scalar:   bool) -> i64 {

    let wrap_scalar: bool = wrap_scalar.unwrap_or(true);

    todo!();
        /*
            return maybe_wrap_dim(dim, dim_post_expr, wrap_scalar);
        */
}

#[inline] pub fn maybe_wrap_dim_with_ptensor(
        dim:    i64,
        tensor: *mut TensorImpl) -> i64 {
    
    todo!();
        /*
            return maybe_wrap_dim(dim, tensor->dim());
        */
}

#[inline] pub fn maybe_wrap_dim_with_tensor_list(
    dim:     i64,
    tensors: TensorList) -> i64 {
    
    todo!();
        /*
            if (tensors.size() == 0) {
        // can't wrap empty TensorList; rely on underlying implementation to throw error if necessary.
        return dim;
      }
      return maybe_wrap_dim(dim, tensors[0].dim());
        */
}

#[inline] pub fn maybe_wrap_dim_with_tensor_sizes(
        dim:          i64,
        tensor_sizes: &Vec<Vec<i64>>) -> i64 {
    
    todo!();
        /*
            if (tensor_sizes.size() == 0) {
        // can't wrap empty list; rely on underlying implementation to throw error if necessary
        return dim;
      }
      return maybe_wrap_dim(dim, tensor_sizes[0].size());
        */
}

/**
  | wrap each dim in the dims array, taking
  | dim_post_expr as the true number of
  | dimensions
  |
  */
#[inline] pub fn maybe_wrap_dims_n(
    dims:          *mut i64,
    ndims:         i64,
    dim_post_expr: i64)  {
    
    todo!();
        /*
            if (dim_post_expr <= 0) {
        dim_post_expr = 1; // this will make range [-1, 0]
      }
      i64 min = -dim_post_expr;
      i64 max = dim_post_expr - 1;
      for (i64 i = 0; i < ndims; ++i) {
        auto &dim = dims[i];
        if (dim < min || dim > max) {
          TORCH_CHECK_INDEX(false,
            "Dimension out of range (expected to be in range of [",
            min, ", ", max, "], but got ", dim, ")");
        }
        if (dim < 0) dim += dim_post_expr;
      }
        */
}

/**
  | Wrap each dim in a contiguous container, taking
  | dim_post_expr as the true number of dimensions
  |
  | E.g. could also be array or SmallVector
  |
  */
#[inline] pub fn maybe_wrap_dims<Container>(
    dims:          &mut Container,
    dim_post_expr: i64)  {

    todo!();
        /*
            return maybe_wrap_dims_n(dims.data(), dims.size(), dim_post_expr);
        */
}

/**
  | previously, size [0] tensors were the only
  | possible empty tensors; thus, it wasn't
  | possible to cat empty tensors unless all the
  | other tensors were 1-dimensional, so we allowed
  | these tensors to be "skipped" (both for wrap
  | dimension behavior and dimension size
  | checking).
  |
  | We maintain this behavior for backwards
  | compatibility, but only for this specific size
  |
  | (i.e. other empty sizes are not skipped).
  |
  */
#[inline] pub fn legacy_cat_wrap_dim(
    dim:          i64,
    tensor_sizes: &Vec<Vec<i64>>) -> i64 {
    
    todo!();
        /*
            for (auto& sizes : tensor_sizes) {
        if (sizes == vector<i64>({0})) {
          continue;
        }
        return maybe_wrap_dim(dim, sizes.size());
      }
      return dim;
        */
}

#[inline] pub fn legacy_cat_wrap_dim_with_tensors(
    dim:     i64,
    tensors: TensorList) -> i64 {
    
    todo!();
        /*
            for (auto& tensor : tensors) {
        if (tensor.dim() == 1 && tensor.sizes()[0] == 0) {
          continue;
        }
        return maybe_wrap_dim(dim, tensor.dim());
      }
      return dim;
        */
}

/// wrap negative dims in a vector
///
#[inline] pub fn wrap_all_dims(
        dims_to_wrap:      &mut Vec<i64>,
        tensor_total_dims: i64)  {
    
    todo!();
        /*
            for (usize i = 0; i < dims_to_wrap.size(); i++) {
        dims_to_wrap[i] = maybe_wrap_dim(dims_to_wrap[i], tensor_total_dims);
      }
        */
}
