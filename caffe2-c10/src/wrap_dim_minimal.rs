crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/WrapDimMinimal.h]

#[inline] pub fn maybe_wrap_dim(
    dim:           i64,
    dim_post_expr: i64,
    wrap_scalar:   Option<bool>) -> i64 {

    let wrap_scalar: bool = wrap_scalar.unwrap_or(true);

    todo!();
        /*
            if (dim_post_expr <= 0) {
        if (!wrap_scalar) {
          TORCH_CHECK_INDEX(
              false,
              "dimension specified as ",
              dim,
              " but tensor has no dimensions");
        }
        dim_post_expr = 1; // this will make range [-1, 0]
      }

      int64_t min = -dim_post_expr;
      int64_t max = dim_post_expr - 1;
      if (dim < min || dim > max) {
        TORCH_CHECK_INDEX(
            false,
            "Dimension out of range (expected to be in range of [",
            min,
            ", ",
            max,
            "], but got ",
            dim,
            ")");
      }
      if (dim < 0)
        dim += dim_post_expr;
      return dim;
        */
}
