crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/LerpKernel.cpp]

pub fn lerp_kernel_scalar(
        ret:    &mut Tensor,
        self_:  &Tensor,
        end:    &Tensor,
        weight: &Scalar)  {
    
    todo!();
        /*
            TORCH_CHECK(self.dtype() == end.dtype(), "expected dtype ", self.dtype(), " for `end` but got dtype ", end.dtype());
      auto iter = TensorIterator::borrowing_binary_op(ret, self, end);
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(ret.scalar_type(), "lerp_kernel_scalar", [&] {
        using Value = typename scalar_Valueype<Scalar>::type;
        Scalar weight_val = weight.to<Scalar>();
        native::cpu_kernel(
            iter,
            [weight_val](Scalar self_val, Scalar end_val) {
              return (zabs<Scalar, Value>(weight_val) < 0.5)
                  ? self_val + weight_val * (end_val - self_val)
                  : end_val - (end_val - self_val) * (Scalar(1) - weight_val);
            });
      });
        */
}

pub fn lerp_kernel_tensor(
        ret:     &mut Tensor,
        self_:   &Tensor,
        end:     &Tensor,
        weights: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(self.dtype() == end.dtype(), "expected dtype ", self.dtype(), " for `end` but got dtype ", end.dtype());
      TORCH_CHECK(self.dtype() == weights.dtype(), "expected dtype ", self.dtype(), " for `weights` but got dtype ", weights.dtype());
      auto iter = TensorIteratorConfig()
        .add_output(ret)
        .add_input(self)
        .add_input(end)
        .add_input(weights)
        .build();
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(ret.scalar_type(), "lerp_kernel_tensor", [&] {
        using Value = typename scalar_Valueype<Scalar>::type;
        native::cpu_kernel(
            iter,
            [](Scalar self_val, Scalar end_val, Scalar weight_val) {
              return (zabs<Scalar, Value>(weight_val) < 0.5)
                  ? self_val + weight_val * (end_val - self_val)
                  : end_val - (end_val - self_val) * (Scalar(1) - weight_val);
            });
      });
        */
}

register_dispatch!{lerp_kernel_scalar_weight, &lerp_kernel_scalar}
register_dispatch!{lerp_kernel_tensor_weight, &lerp_kernel_tensor}
