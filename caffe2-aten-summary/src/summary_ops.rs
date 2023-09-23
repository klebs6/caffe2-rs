// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/SummaryOps.cpp]

/**
  | Returns the frequency of elements of
  | input non-negative integer tensor.
  |
  */
pub fn bincount_cpu_template<input_t, weights_t>(
    self_:     &Tensor,
    weights:   &Tensor,
    minlength: i64) -> Tensor {

    todo!();
        /*
            if (minlength < 0) {
        AT_ERROR("minlength should be >= 0");
      }
      if (self.dim() == 1 && self.numel() == 0) {
        return native::zeros({minlength}, kLong);
      }
      if (self.dim() != 1 || *self.min().data_ptr<input_t>() < 0) {
        AT_ERROR("bincount only supports 1-d non-negative integral inputs.");
      }

      bool has_weights = weights.defined();
      if (has_weights && weights.size(0) != self.size(0)) {
        AT_ERROR("input and weights should have the same length");
      }

      Tensor output;
      i64 self_size = self.size(0);
      i64 nbins = static_cast<i64>(*self.max().data_ptr<input_t>()) + 1L;
      nbins = max(nbins, minlength); // at least minlength # of bins

      const input_t* self_p = self.data_ptr<input_t>();
      if (has_weights) {
        output = native::zeros(
            {nbins},
            optTypeMetaToScalarType(weights.options().dtype_opt()),
            weights.options().layout_opt(),
            weights.options().device_opt(),
            weights.options().pinned_memory_opt());
        weights_t* output_p = output.data_ptr<weights_t>();
        const weights_t* weights_p = weights.data_ptr<weights_t>();
        for (i64 i = 0; i < self_size; i++) {
          output_p[self_p[i]] += weights_p[i];
        }
      } else {
        output = native::zeros({nbins}, kLong);
        i64* output_p = output.data_ptr<i64>();
        for (i64 i = 0; i < self_size; i++) {
          output_p[self_p[i]] += 1L;
        }
      }
      return output;
        */
}

pub fn bincount_cpu(
    self_:       &Tensor,
    weights_opt: &Option<Tensor>,
    minlength:   i64) -> Tensor {

    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weights_maybe_owned = borrow_from_optional_tensor(weights_opt);
      const Tensor& weights = *weights_maybe_owned;

      return AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "bincount_cpu", [&] {
        const auto scalar = weights.scalar_type();
        if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
          return _bincount_cpu_template<Scalar, float>(self.contiguous(), weights.contiguous(), minlength);
        return _bincount_cpu_template<Scalar, double>(
            self.contiguous(), weights.contiguous().to(kDouble), minlength);
      });
        */
}
