crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/UnaryOps.cpp]

#[cfg(not(AT_MKLDNN_ENABLED))]
pub use not_mkldnn::*;

#[cfg(AT_MKLDNN_ENABLED)]
pub use mkldnn::*;

#[cfg(not(AT_MKLDNN_ENABLED))]
mod not_mkldnn {
    use super::*;

    pub fn mkldnn_sigmoid<'a>(self_: &Tensor) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_sigmoid: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_sigmoid_mut(self_: &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_sigmoid_: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_tanh<'a>(self_: &Tensor) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_tanh: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_tanh_mut(self_: &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_tanh_: ATen not compiled with MKLDNN support");
            */
    }
}

#[cfg(AT_MKLDNN_ENABLED)]
mod mkldnn {
    use super::*;

    pub fn mkldnn_sigmoid<'a>(self_: &Tensor) -> Tensor {
        
        todo!();
            /*
                ideep::tensor& x = itensor_from_mkldnn(self);
          ideep::tensor y;
          ideep::eltwise_forward::compute(
              x, y, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
          return new_with_itensor_mkldnn(move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                         self.options().device_opt());
            */
    }

    pub fn mkldnn_sigmoid_mut(self_: &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                ideep::tensor& x = itensor_from_mkldnn(self);
          ideep::eltwise_forward::compute(
              x, x, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
          return self;
            */
    }

    pub fn mkldnn_tanh<'a>(self_: &Tensor) -> Tensor {
        
        todo!();
            /*
                ideep::tensor& x = itensor_from_mkldnn(self);
          ideep::tensor y;
          ideep::eltwise_forward::compute(
              x, y, ideep::algorithm::eltwise_tanh, ideep::prop_kind::forward);
          return new_with_itensor_mkldnn(move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                         self.options().device_opt());
            */
    }

    pub fn mkldnn_tanh_mut(self_: &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                ideep::tensor& x = itensor_from_mkldnn(self);
          ideep::eltwise_forward::compute(
              x, x, ideep::algorithm::eltwise_tanh, ideep::prop_kind::forward);
          return self;
            */
    }
}
