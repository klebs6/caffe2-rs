crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/SoftMax.cpp]

#[cfg(not(AT_MKLDNN_ENABLED))]
pub fn mkldnn_softmax(
        self_:         &Tensor,
        dim:           i64,
        half_to_float: bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false, "mkldnn_softmax: ATen not compiled with MKLDNN support");
        */
}

#[cfg(AT_MKLDNN_ENABLED)]
pub fn mkldnn_softmax(
        self_:         &Tensor,
        dim:           i64,
        half_to_float: bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          !half_to_float,
          "softmax with half to float conversion is not supported on Mkldnn");
      const i64 wrapped_dim = maybe_wrap_dim(dim, self.dim());
      ideep::tensor& x = itensor_from_mkldnn(self);
      ideep::tensor y;
      ideep::softmax_forward::compute(x, y, wrapped_dim);
      return new_with_itensor_mkldnn(move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                     self.options().device_opt());
        */
}
