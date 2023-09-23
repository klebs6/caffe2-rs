crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/tensor_operators.cpp]

/**
  | All comparator operators will be named
  | "<aten op name>_quantized_cpu". '_out'
  | will be appended for the 'out' variant
  | of the op.
  | 
  | TODO: This is an inefficient implementation
  | that uses `.dequantize`.
  | 
  | Need a more efficient implementation.
  |
  */
#[macro_export] macro_rules! define_comparator {
    ($at_op:ident) => {
        /*
        
        Tensor& at_op##_out_quantized_cpu(const Tensor& self, 
                                        const Scalar& other, Tensor& out) { 
          TORCH_CHECK(out.dtype() == ScalarType::Bool, 
                      "The 'out' tensor must have dtype 'torch.bool'"); 
          auto self_dq = self.dequantize(); 
          return  at_op##_out(out, self_dq, other); 
        } 
        Tensor at_op##_quantized_cpu(const Tensor& self, const Scalar& other) { 
          auto self_dq = self.dequantize(); 
          return  at_op(self_dq, other); 
        } 
        Tensor& at_op##_out_quantized_cpu(const Tensor& self, 
                                        const Tensor& other, Tensor& out) { 
          /* We infer size to make sure the tensors are compatible. */
          infer_size_dimvector(self.sizes(), other.sizes()); 
          TORCH_CHECK(out.dtype() == ScalarType::Bool, 
                      "The 'out' tensor must have dtype 'torch.bool'"); 
          auto self_dq = self.dequantize(); 
          auto other_dq = other.dequantize(); 
          return  at_op##_out(out, self_dq, other_dq); 
        } 
        Tensor at_op##_quantized_cpu(const Tensor& self, const Tensor& other) { 
          /* We infer size to make sure the tensors are compatible. */
          infer_size_dimvector(self.sizes(), other.sizes()); 
          auto self_dq = self.dequantize(); 
          auto other_dq = other.dequantize(); 
          return  at_op(self_dq, other_dq); 
        }
        */
    }
}

#[macro_export] macro_rules! at_forall_operators {
    ($_:ident) => {
        /*
        
        _(ne)                          
        _(eq)                          
        _(ge)                          
        _(le)                          
        _(gt)                          
        _(lt)                          
        */
    }
}

lazy_static!{
    /*
    at_forall_operators!{define_comparator}
    */
}

pub fn quantized_resize_cpu<'a>(
    self_:                  &Tensor,
    size:                   &[i32],
    optional_memory_format: Option<MemoryFormat>) -> &'a Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          !optional_memory_format.has_value(),
          "Unsupported memory format for quantized tensor resize ",
          optional_memory_format.value());
      auto qscheme = self.quantizer()->qscheme();
      TORCH_CHECK(
          qscheme == QScheme::PER_TENSOR_AFFINE ||
              qscheme == QScheme::PER_TENSOR_SYMMETRIC,
          "Can only resize quantized tensors with per-tensor schemes!");
      auto* self_ = self.unsafeGetTensorImpl();
      resize_impl_cpu_(self_, size, /*strides=*/nullopt);
      return self;
        */
}
