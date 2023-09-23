crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/TensorShape.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/TensorShape.cpp]

#[cfg(not(AT_MKLDNN_ENABLED))]
pub fn mkldnn_view(
        self_: &Tensor,
        size:  &[i32]) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false, "mkldnn_reshape: ATen not compiled with MKLDNN support");
        */
}


#[cfg(not(AT_MKLDNN_ENABLED))]
pub fn mkldnn_reshape(
        self_: &Tensor,
        size:  &[i32]) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false, "mkldnn_reshape: ATen not compiled with MKLDNN support");
        */
}


#[cfg(not(AT_MKLDNN_ENABLED))]
pub fn mkldnn_clone(
        self_:                  &Tensor,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false, "mkldnn_clone: ATen not compiled with MKLDNN support");
        */
}

#[cfg(not(AT_MKLDNN_ENABLED))]
pub fn mkldnn_transpose(
        self_: &Tensor,
        dim0:  i64,
        dim1:  i64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false, "mkldnn_transpose: ATen not compiled with MKLDNN support");
        */
}

#[cfg(not(AT_MKLDNN_ENABLED))]
pub fn mkldnn_transpose_mut<'a>(
        self_: &mut Tensor,
        dim0:  i64,
        dim1:  i64) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false, "mkldnn_transpose_: ATen not compiled with MKLDNN support");
        */
}

#[cfg(AT_MKLDNN_ENABLED)]
pub fn mkldnn_view(
        self_: &Tensor,
        size:  &[i32]) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false,
          "Currently Mkldnn tensor does not support view. Change to use reshape instead");
        */
}

#[cfg(AT_MKLDNN_ENABLED)]
pub fn mkldnn_reshape(
        self_: &Tensor,
        size:  &[i32]) -> Tensor {
    
    todo!();
        /*
            auto inferred_size = infer_size(size, self.numel());
      if (self.sizes() == inferred_size) {
        return self;
      }
      const ideep::tensor& x = itensor_from_mkldnn(self);
      ideep::tensor y{x};
      y.reshape(inferred_size);
      return new_with_itensor_mkldnn(move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                     self.options().device_opt());
        */
}

#[cfg(AT_MKLDNN_ENABLED)]
pub fn mkldnn_clone(
        self_:                  &Tensor,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          !optional_memory_format.has_value(),
          "unsupported memory format option ",
          optional_memory_format.value());
      ideep::tensor& src = itensor_from_mkldnn(self);
      ideep::tensor dst;
      ideep::direct_copy::compute(src, dst);
      return new_with_itensor_mkldnn(move(dst), optTypeMetaToScalarType(self.options().dtype_opt()),
                                     self.options().device_opt());
        */
}

#[cfg(AT_MKLDNN_ENABLED)]
pub fn mkldnn_transpose(
        self_: &Tensor,
        dim0:  i64,
        dim1:  i64) -> Tensor {
    
    todo!();
        /*
            const ideep::tensor& x = itensor_from_mkldnn(self);
      ideep::tensor y;
      vector<int> axes(x.ndims());
      iota(axes.begin(), axes.end(), 0);
      swap(axes[dim0], axes[dim1]);
      y.transpose_from(x, axes);
      return new_with_itensor_mkldnn(move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                     self.options().device_opt());
        */
}

#[cfg(AT_MKLDNN_ENABLED)]
pub fn mkldnn_transpose_mut<'a>(
        self_: &mut Tensor,
        dim0:  i64,
        dim1:  i64) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false, "mkldnn_transpose_: in-place mkldnn operations are not supported yet");
        */
}
