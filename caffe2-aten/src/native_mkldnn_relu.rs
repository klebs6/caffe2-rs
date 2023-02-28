crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/Relu.cpp]

#[cfg(not(AT_MKLDNN_ENABLED))]
pub use not_mkldnn::*;

#[cfg(AT_MKLDNN_ENABLED)]
pub use mkldnn::*;

#[cfg(not(AT_MKLDNN_ENABLED))]
mod not_mkldnn {
    use super::*;

    pub fn mkldnn_relu(input: &Tensor) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_relu: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_relu_mut(input: &mut Tensor) -> &mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_relu_: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_relu_backward(
            grad_output: &Tensor,
            input:       &Tensor,
            threshold:   &Scalar) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_relu_backward: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_gelu(input: &Tensor) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_gelu: ATen not compiled with MKLDNN support");
            */
    }
}

#[cfg(AT_MKLDNN_ENABLED)]
mod mkldnn {
    use super::*;

    pub fn mkldnn_relu(input: &Tensor) -> Tensor {
        
        todo!();
            /*
                if (input.scalar_type() == ScalarType::BFloat16) {
            TORCH_CHECK(mkldnn_bf16_device_check(),
                "mkldnn_relu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
          }

          const ideep::tensor& x = itensor_from_mkldnn(input);
          ideep::tensor y;
          ideep::eltwise_forward::compute(
              x, y, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
          return new_with_itensor_mkldnn(move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                         input.options().device_opt());
            */
    }

    pub fn mkldnn_relu_mut(input: &mut Tensor) -> &mut Tensor {
        
        todo!();
            /*
                if (input.scalar_type() == ScalarType::BFloat16) {
            TORCH_CHECK(mkldnn_bf16_device_check(),
                "mkldnn_relu_: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
          }

          ideep::tensor& x = itensor_from_mkldnn(input);
          ideep::eltwise_forward::compute(
              x, x, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
          return input;
            */
    }

    pub fn mkldnn_relu_backward(
            grad_output: &Tensor,
            input:       &Tensor,
            threshold:   &Scalar) -> Tensor {
        
        todo!();
            /*
                ideep::tensor& x = itensor_from_mkldnn(input);
          ideep::tensor grady = itensor_from_mkldnn(grad_output);
          ideep::tensor gradx;
          ideep::eltwise_backward::compute(x, grady, gradx,
              ideep::algorithm::eltwise_relu, /*alpha*/ 0.0);
          return new_with_itensor_mkldnn(move(gradx),
                                         optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                         grad_output.options().device_opt());
            */
    }

    pub fn mkldnn_gelu(input: &Tensor) -> Tensor {
        
        todo!();
            /*
                if (input.scalar_type() == ScalarType::BFloat16) {
            TORCH_CHECK(mkldnn_bf16_device_check(),
                "mkldnn_gelu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
          }

          const ideep::tensor& x = itensor_from_mkldnn(input);
          ideep::tensor y;
          ideep::eltwise_forward::compute(
              x, y, ideep::algorithm::eltwise_gelu_erf, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
          return new_with_itensor_mkldnn(move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                         input.options().device_opt());
            */
    }
}
