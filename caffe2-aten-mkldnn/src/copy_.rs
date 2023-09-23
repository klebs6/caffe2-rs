crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/Copy.cpp]

#[cfg(not(AT_MKLDNN_ENABLED))]
pub fn copy_mkldnn<'a>(
        self_:        &mut Tensor,
        src:          &Tensor,
        non_blocking: bool) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false, "copy_mkldnn_: ATen not compiled with MKLDNN support");
        */
}

#[cfg(AT_MKLDNN_ENABLED)]
pub fn copy_mkldnn<'a>(
        self_:        &mut Tensor,
        src:          &Tensor,
        non_blocking: bool) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          self.sizes() == src.sizes(),
          "copy_mkldnn_: only support same size tensor.");
      TORCH_CHECK(
          self.is_mkldnn() && src.is_mkldnn(),
          "copy_mkldnn_: between mkldnn layout and dense Tensors is not implemented! Found self type = ",
          self.toString(),
          " and src type = ",
          src.toString());
      ideep::tensor& x = itensor_from_mkldnn(src);
      ideep::tensor& y = itensor_from_mkldnn(self);
      ideep::direct_copy::compute(x, y);
      return self;
        */
}
