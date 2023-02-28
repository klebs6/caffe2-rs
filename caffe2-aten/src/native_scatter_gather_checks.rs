crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ScatterGatherChecks.h]

/**
  | checks whether index.dtype == int64
  | and self.dtyp == src.dtype if src is
  | a Tensor
  |
  */
pub fn scatter_gather_dtype_check(
        method_name: &String,
        self_:       &Tensor,
        index:       &Tensor,
        src_opt:     &Option<Tensor>)  {

    todo!();
        /*
            TORCH_CHECK(
        index.scalar_type() == ScalarType::Long,
        method_name, "(): Expected dtype int64 for index"
      );

      if (src_opt.has_value()) {
        auto src = src_opt.value();
        TORCH_CHECK(
          self.scalar_type() == src.scalar_type(),
          method_name, "(): Expected self.dtype to be equal to src.dtype"
        );
      }
        */
}

/**
  | Used for `gather`-like methods
  |
  | Test:
  |
  | 1. index.size(d) == self.size(d) for all d != dim
  |
  | 2. index.size(d) <= src.size(d) for all d != dim
  |
  | 3. index.dim() == self.dim() == src.dim()
  |
  */
pub fn gather_shape_check(
    self_: &Tensor,
    dim:   i64,
    index: &Tensor,
    src:   &Tensor)  {
    
    todo!();
        /*
            auto self_dims = ensure_nonempty_dim(self.dim());
      TORCH_CHECK(self_dims == ensure_nonempty_dim(index.dim()),
        "Index tensor must have the same number of dimensions as out tensor"
      );

      auto src_dims = ensure_nonempty_dim(src.dim());
      TORCH_CHECK(src_dims == ensure_nonempty_dim(index.dim()),
        "Index tensor must have the same number of dimensions as input tensor"
      );

      for (i64 i = 0; i < self_dims; ++i) {
        if (i != dim) {
          TORCH_CHECK(
            ensure_nonempty_size(index, i) == ensure_nonempty_size(self, i),
            "Size does not match at dimension ", i,
            " get ", ensure_nonempty_size(self, i),
            " vs ", ensure_nonempty_size(index, i)
          );

          TORCH_CHECK(
            ensure_nonempty_size(index, i) <= ensure_nonempty_size(src, i),
            "Size does not match at dimension ", i,
            " expected index ", index.sizes(),
            " to be smaller than src ", src.sizes(),
            " apart from dimension ", dim
          );
        }
      }
        */
}

/**
  | Used for `scatter` and `scatter_add`
  |
  | Tests:
  |
  |  1. index.size(d) <= self.size(d) for all d != dim
  |
  |  2. index.size(d) <= src.size(d) for all d if src is a Tensor
  |
  |  3. index.dim() == self.dim() == src.dim()
  |
  */
pub fn scatter_shape_check(
        self_:   &Tensor,
        dim:     i64,
        index:   &Tensor,
        src_opt: &Option<Tensor>)  {

    todo!();
        /*
            if (index.numel() == 0) {
        return;
      }

      TORCH_CHECK(
        ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
        "Index tensor must have the same number of dimensions as self tensor"
      );

      bool is_wrong_shape = false;
      i64 self_dims = ensure_nonempty_dim(self.dim());

      //  Check: index.size(d) <= self.size(d) for all d != dim
      for (i64 d = 0; d < self_dims; ++d) {
        i64 index_d_size = ensure_nonempty_size(index, d);
        if (d == dim) continue;
        if (index_d_size > ensure_nonempty_size(self, d)) {
          is_wrong_shape = true;
          break;
        }
      }

      //  Check: index.size(d) <= src.size(d) for all d if src is Tensor
      if (!is_wrong_shape && src_opt.has_value()) {
        auto src = src_opt.value();
        for (i64 d = 0; d < self_dims; ++d) {
          i64 index_d_size = ensure_nonempty_size(index, d);
          if (index_d_size > ensure_nonempty_size(src, d)) {
            is_wrong_shape = true;
            break;
          }
        }
      }

      if (src_opt.has_value()) {
        auto src = src_opt.value();

        TORCH_CHECK(
          ensure_nonempty_dim(src.dim()) == ensure_nonempty_dim(index.dim()),
          "Index tensor must have the same number of dimensions as src tensor"
        );

        TORCH_CHECK(!is_wrong_shape,
          "Expected index ", index.sizes(),
          " to be smaller than self ", self.sizes(),
          " apart from dimension ", dim,
          " and to be smaller size than src ", src.sizes()
        );
      }
      else {
        TORCH_CHECK(!is_wrong_shape,
          "Expected index ", index.sizes(),
          " to be smaller than self ", self.sizes(),
          " apart from dimension ", dim
        );
      }
        */
}
