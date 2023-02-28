crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/Copy.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/Copy.cpp]

/**
  | Copying from float to QInt, used for
  | assigning float value to QTensor
  |
  */
pub fn quantized_copy_from_float_cpu(
        self_: &mut Tensor,
        src:   &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          src.scalar_type() == kFloat,
          "Quantized copy only works with kFloat as source Tensor");
      TORCH_CHECK(
          self.is_contiguous() && src.is_contiguous(),
          "Quantized copy only works with contiguous Tensors");
      TORCH_CHECK(
          self.sizes().equals(src.sizes()),
          "Quantized copy only works with Tensors with the same shape");
      TORCH_CHECK(
          self.device().type() == kCPU,
          "Quantized copy only works with QuantizedCPU Tensors");
      AT_DISPATCH_QINT_TYPES(self.scalar_type(), "Copy", [&]() {
        float* src_data = src.data_ptr<float>();
        Scalar* self_data = self.data_ptr<Scalar>();
        for (int i = 0; i < self.numel(); ++i) {
          self_data[i] = quantize_val<Scalar>(
              self.q_scale(), self.q_zero_point(), src_data[i]);
        }
      });
      return self;
        */
}
