crate::ix!();

#[inline] pub fn check_args(
        row:        i64,
        col:        i64,
        layout_opt: Option<Layout>)  {
    
    todo!();
        /*
            TORCH_CHECK(row >= 0, "row must be non-negative, got", row);
      TORCH_CHECK(col >= 0, "col must be non-negative, got", col);
      if (layout_opt.has_value()) {
        TORCH_CHECK(
          *layout_opt == kStrided,
          "only support layout=torch.strided, got",
          *layout_opt)
      }
        */
}

/**
  | assumes maximum value in created tensor
  | is n-1 (e.g., torch.randperm(n))
  |
  */
#[inline] pub fn check_supported_max_int_with_precision(
        n:      i64,
        tensor: &Tensor)  {
    
    todo!();
        /*
            // match defined() to behavior of checks below
      TORCH_CHECK(scalar_tensor(n>0?n-1:n, tensor.options()).defined(),
                  "n is too large for result tensor type: '", tensor.toString(), "'");

      // Ensure sufficient precision for floating point representation.
      switch (tensor.scalar_type()) {
        case ScalarType::Half:
          TORCH_CHECK(n <= (i64(1) << 11) + 1, "n cannot be greater than 2049 for Half type.");
          break;
        case ScalarType::Float:
          TORCH_CHECK(n <= (i64(1) << 24) + 1, "n cannot be greater than 2^24+1 for Float type.");
          break;
        case ScalarType::Double:  // Unlikely to happen, but doesn't hurt to check
          TORCH_CHECK(n <= (i64(1) << 53) + 1, "n cannot be greater than 2^53+1 for Double type.");
          break;
        default:
          break;
      }
        */
}
