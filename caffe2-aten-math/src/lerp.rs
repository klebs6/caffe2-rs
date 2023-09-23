crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Lerp.h]

pub type LerpFnScalar = fn(
        ret:    &mut Tensor,
        self_:  &Tensor,
        end:    &Tensor,
        weight: &Scalar
) -> c_void;

pub type LerpFnTensor = fn(
        ret:     &mut Tensor,
        self_:   &Tensor,
        end:     &Tensor,
        weights: &Tensor
) -> c_void;

declare_dispatch!{lerp_fn_scalar, lerp_kernel_scalar_weight}
declare_dispatch!{lerp_fn_tensor, lerp_kernel_tensor_weight}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Lerp.cpp]

pub fn lerp_cpu_tensor_out<'a>(
        self_:  &Tensor,
        end:    &Tensor,
        weight: &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            lerp_kernel_tensor_weight(kCPU, result, self, end, weight);
      return result;
        */
}

pub fn lerp_cpu_scalar_out<'a>(
        self_:  &Tensor,
        end:    &Tensor,
        weight: &Scalar,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            lerp_kernel_scalar_weight(kCPU, result, self, end, weight);
      return result;
        */
}

pub fn lerp_cpu_tensor_a<'a>(
        self_:  &mut Tensor,
        end:    &Tensor,
        weight: &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            lerp_kernel_tensor_weight(kCPU, self, self, end, weight);
      return self;
        */
}

pub fn lerp_cpu_scalar_a<'a>(
        self_:  &mut Tensor,
        end:    &Tensor,
        weight: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            lerp_kernel_scalar_weight(kCPU, self, self, end, weight);
      return self;
        */
}

pub fn lerp_cpu_tensor_b(
        self_:  &Tensor,
        end:    &Tensor,
        weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      lerp_kernel_tensor_weight(kCPU, result, self, end, weight);
      return result;
        */
}

pub fn lerp_cpu_scalar_b(
        self_:  &Tensor,
        end:    &Tensor,
        weight: &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      lerp_kernel_scalar_weight(kCPU, result, self, end, weight);
      return result;
        */
}

define_dispatch!{lerp_kernel_scalar_weight}
define_dispatch!{lerp_kernel_tensor_weight}
