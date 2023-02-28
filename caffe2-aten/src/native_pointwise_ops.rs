/*!
  | Ternary and higher-order pointwise
  | operations
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/PointwiseOps.h]

lazy_static!{
    /*
    using pointwise_fn = void (*)(TensorIterator&, const Scalar& scalar);
    using pointwise_fn_double = void (*)(TensorIterator&, const Scalar&, double);
    */
}

declare_dispatch!{pointwise_fn, addcmul_stub}
declare_dispatch!{pointwise_fn, addcdiv_stub}
declare_dispatch!{pointwise_fn_double, smooth_l1_backward_stub}
declare_dispatch!{pointwise_fn_double, huber_backward_stub}
declare_dispatch!{pointwise_fn, mse_backward_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/PointwiseOps.cpp]
// Ternary and higher-order pointwise operations


pub fn addcmul_a(
        self_:   &Tensor,
        tensor1: &Tensor,
        tensor2: &Tensor,
        value:   &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return addcmul_out(result, self, tensor1, tensor2, value);
        */
}

pub fn addcmul_b(
        self_:   &mut Tensor,
        tensor1: &Tensor,
        tensor2: &Tensor,
        value:   &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return addcmul_out(self, self, tensor1, tensor2, value);
        */
}

pub fn addcmul_out(
        self_:   &Tensor,
        tensor1: &Tensor,
        tensor2: &Tensor,
        value:   &Scalar,
        result:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            checkBackend("addcmul_cpu", result, self.options().backend());
      auto iter = TensorIteratorConfig()
        .add_output(result)
        .add_input(self)
        .add_input(tensor1)
        .add_input(tensor2)
        .build();
      addcmul_stub(iter.device_type(), iter, value);
      return result;
        */
}

pub fn addcdiv_a(
        self_:   &Tensor,
        tensor1: &Tensor,
        tensor2: &Tensor,
        value:   &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return addcdiv_out(result, self, tensor1, tensor2, value);
        */
}


pub fn addcdiv_b(
        self_:   &mut Tensor,
        tensor1: &Tensor,
        tensor2: &Tensor,
        value:   &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return addcdiv_out(self, self, tensor1, tensor2, value);
        */
}

pub fn addcdiv_out(
        self_:   &Tensor,
        tensor1: &Tensor,
        tensor2: &Tensor,
        value:   &Scalar,
        result:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            if (isIntegralType(tensor1.scalar_type(), /*includeBool=*/ true)
          && isIntegralType(tensor2.scalar_type(), /*includeBool=*/ true)) {
        TORCH_CHECK(false,
          "Integer division with addcdiv is no longer supported, and in a future  ",
          "release addcdiv will perform a true division of tensor1 and tensor2. ",
          "The historic addcdiv behavior can be implemented as ",
          "(input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) ",
          "for integer inputs and as ",
          "(input + value * tensor1 / tensor2) for float inputs. ",
          "The future addcdiv behavior is just the latter implementation: ",
          "(input + value * tensor1 / tensor2), for all dtypes.");
      }
      checkBackend("addcdiv_cpu", result, self.options().backend());
      auto iter = TensorIteratorConfig()
        .add_output(result)
        .add_input(self)
        .add_input(tensor1)
        .add_input(tensor2)
        .build();
      addcdiv_stub(iter.device_type(), iter, value);
      return result;
        */
}

define_dispatch!{addcmul_stub}
define_dispatch!{addcdiv_stub}
