crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qsort.cpp]

/**
  | Currently internal-only.
  |
  | This implementation assumes the quantizer for
  | the input and the out- put are the same.
  |
  | If we want to support this publicly, we need to
  | add a requantization step to the kernel.
  */
pub fn quantized_topk_out_cpu<'a>(
    values:  &mut Tensor,
    indices: &mut Tensor,
    self_:   &Tensor,
    k:       i64,
    dim:     i64,
    largest: bool,
    sorted:  bool) -> (&'a mut Tensor,&'a mut Tensor) {

    todo!();
        /*
            i64 dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
      TORCH_CHECK(
          k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
          "selected index k out of range");
      _allocate_or_resize_output_with_indices(values, indices, self, dim_, k);

      qtopk_stub(kCPU, values, indices, self, k, dim, largest, sorted);

      return forward_as_tuple(values, indices);
        */
}

pub fn topk_quantized_cpu(
        self_:   &Tensor,
        k:       i64,
        dim:     i64,
        largest: bool,
        sorted:  bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            auto qscheme = self.qscheme();
      TORCH_CHECK(
          qscheme == QScheme::PER_TENSOR_AFFINE ||
              qscheme == QScheme::PER_TENSOR_SYMMETRIC,
          "Top-K is only supported on per-tensor quantization");
      Tensor values = _empty_affine_quantized(
        {0},
        self.options(),
        self.q_scale(),
        self.q_zero_point());
      Tensor indices = empty({0}, self.options().dtype(kLong));
      return quantized_topk_out_cpu(values, indices, self, k, dim, largest, sorted);
        */
}

define_dispatch!{qtopk_stub}
