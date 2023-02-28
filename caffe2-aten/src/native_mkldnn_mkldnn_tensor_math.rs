crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/MkldnnTensorMath.cpp]

#[cfg(not(AT_MKLDNN_ENABLED))]
pub fn mkldnn_zero(self_: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false, "mkldnn_zero_: ATen not compiled with MKLDNN support");
        */
}

#[cfg(AT_MKLDNN_ENABLED)]
pub fn mkldnn_zero(self_: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            using Vec = vec::Vectorized<float>;

      ideep::tensor& x = itensor_from_mkldnn(self);

      auto n = x.get_nelems();
      auto* x_ = static_cast<float*>(x.get_data_handle());
      parallel_for(0, n, 2048, [x_](i64 begin, i64 end) {
        vec::map(
            [](Vec /* unused */) { return 0.0; },
            x_ + begin,
            x_ + begin,
            end - begin);
      });

      return self;
        */
}
