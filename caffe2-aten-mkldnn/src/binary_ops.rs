crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/BinaryOps.cpp]

#[cfg(target_feature = "mkldnn")]
pub fn empty_binary_op(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            if (!self.requires_grad() && !other.requires_grad()) {
        auto out_size = infer_size(self.sizes(), other.sizes());
        auto out_dtype = promoteTypes(
            c10::typeMetaToScalarType(self.dtype()),
            c10::typeMetaToScalarType(other.dtype()));
        TORCH_CHECK(
            self.device() == other.device(),
            "Expected same device for binary mkldnn op");
        return empty_mkldnn(
            out_size,
            out_dtype,
            self.options().layout_opt(),
            self.options().device_opt(),
            self.options().pinned_memory_opt());
      } else {
        TORCH_CHECK(
            false,
            "MKLDNN does not support Binary Ops with a 0-dimension Tensor in training");
      }
        */
}

#[cfg(target_feature = "mkldnn")]
pub fn mkldnn_add_out<'a>(
        self_:  &Tensor,
        other:  &Tensor,
        alpha:  &Scalar,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            ideep::tensor& x = itensor_from_mkldnn(self);
      ideep::tensor& y = itensor_from_mkldnn(other);

      ideep::tensor& z = itensor_from_mkldnn(result);
      if (result.is_same(other)) {
        const std::vector<float> scales{alpha.to<float>(), 1.0};
        ideep::sum::compute(scales, {y, x}, z);
      } else {
        const std::vector<float> scales{1.0, alpha.to<float>()};
        ideep::sum::compute(scales, {x, y}, z);
      }

      return result;
        */
}

#[cfg(target_feature = "mkldnn")]
pub fn mkldnn_add_mut(
        self_: &Tensor,
        other: &Tensor,
        alpha: &Scalar) -> Tensor {
    
    todo!();
        /*
            if (self.numel() == 0 || other.numel() == 0) {
        return emptyBinaryOp(self, other);
      }

      ideep::tensor& x = itensor_from_mkldnn(self);
      ideep::tensor& y = itensor_from_mkldnn(other);

      ideep::tensor z;
      const std::vector<float> scales{1.0, alpha.to<float>()};
      ideep::sum::compute(scales, {x, y}, z);

      return new_with_itensor_mkldnn(std::move(z), optTypeMetaToScalarType(self.options().dtype_opt()),
                                     self.options().device_opt());
        */
}

#[cfg(target_feature = "mkldnn")]
pub fn mkldnn_add<'a>(
        self_: &mut Tensor,
        other: &Tensor,
        alpha: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::mkldnn_add_out(self, other, alpha, self);
        */
}

#[cfg(target_feature = "mkldnn")]
pub fn mkldnn_mul_out<'a>(
        self_:  &Tensor,
        other:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(result.sizes() == self.sizes(),
                 "mkldnn_mul_out: the output size should be same as input size");
      ideep::tensor& z = itensor_from_mkldnn(result);
      ideep::tensor& x = itensor_from_mkldnn(self);

      // for zero_dim tensor
      if (other.ndimension() == 0) {
        ideep::eltwise_forward::compute(
          x, z, ideep::algorithm::eltwise_linear,
          ideep::prop_kind::forward_inference, /*alpha*/ other.item().to<float>());

        return result;
      } else {
        TORCH_CHECK(self.sizes() == other.sizes(),
                   "mkldnn_mul_out: currently mkldnn not support broadcasting");
        ideep::tensor y = itensor_from_mkldnn(other);
        ideep::binary::compute(x, y, z, dnnl::algorithm::binary_mul);

        return result;
      }
        */
}

#[cfg(target_feature = "mkldnn")]
pub fn mkldnn_mul(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            if (self.numel() == 0 || other.numel() == 0) {
        return emptyBinaryOp(self, other);
      }
      Tensor result = empty_mkldnn(self.sizes(), optTypeMetaToScalarType(self.options().dtype_opt()),
                                   self.options().layout_opt(), self.options().device_opt(),
                                   self.options().pinned_memory_opt());
      return native::mkldnn_mul_out(self, other, result);
        */
}

#[cfg(target_feature = "mkldnn")]
pub fn mkldnn_mul_mut<'a>(
        self_: &mut Tensor,
        other: &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::mkldnn_mul_out(self, other, self);
        */
}

