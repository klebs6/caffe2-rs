crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/make_per_tensor_quantized_tensor.cpp]

pub fn make_per_tensor_quantized_tensor_cpu(
        self_:      &Tensor,
        scale:      f64,
        zero_point: i64) -> Tensor {
    
    todo!();
        /*
            Tensor dst = _empty_affine_quantized(
          self.sizes(),
          self.options().dtype(toQIntType(self.scalar_type())),
          scale,
          zero_point,
          self.suggest_memory_format());
      Tensor self_contig = self.contiguous(self.suggest_memory_format());
      AT_DISPATCH_QINT_TYPES(
          dst.scalar_type(), "make_per_tensor_quantized_tensor", [&]() {
            underlying_t* self_data = self_contig.data_ptr<underlying_t>();
            underlying_t* dst_data =
                reinterpret_cast<underlying_t*>(dst.data_ptr<Scalar>());
            if (self.numel() > 0) {
              memcpy(dst_data, self_data, self.nbytes());
            }
          });
      return dst;
        */
}
