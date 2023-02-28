crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ScalarOps.h]

/**
  | FIXME: this should be (and was)
  | Scalar::toTensor, but there is currently no way
  | to implement this without going through Derived
  | Types (which are not part of core).
  |
  */
#[inline] pub fn scalar_to_tensor(
    s:      &Scalar,
    device: Device) -> Tensor {

    let device: Device = device.unwrap_or(kCPU);

    todo!();
        /*
            // This is the fast track we have for CPU scalar tensors.
      if (device == kCPU) {
        if (s.isFloatingPoint()) {
          return scalar_tensor_static(s, kDouble, kCPU);
        } else if (s.isComplex()) {
          return scalar_tensor_static(s, kComplexDouble, kCPU);
        } else if (s.isBoolean()) {
          return scalar_tensor_static(s, kBool, kCPU);
        } else {
          AT_ASSERT(s.isIntegral(false));
          return scalar_tensor_static(s, kLong, kCPU);
        }
      }
      if (s.isFloatingPoint()) {
        return scalar_tensor(s, device(device).dtype(kDouble));
      } else if (s.isBoolean()) {
        return scalar_tensor(s, device(device).dtype(kBool));
      } else if (s.isComplex()) {
        return scalar_tensor(s, device(device).dtype(kComplexDouble));
      } else {
        AT_ASSERT(s.isIntegral(false));
        return scalar_tensor(s, device(device).dtype(kLong));
      }
        */
}

#[inline] pub fn wrapped_scalar_tensor(
    scalar: &Scalar,
    device: Device) -> Tensor {

    let device: Device = device.unwrap_or(kCPU);

    todo!();
        /*
      auto tensor = scalar_to_tensor(scalar, device);
      tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
      return tensor;
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ScalarOps.cpp]

#[inline] pub fn fill_inplace<Scalar>(
    self_:        &mut Tensor,
    value_scalar: &Scalar)  {

    todo!();
        /*
            auto value = value_scalar.to<Scalar>();
      Scalar* dptr = static_cast<Scalar*>(self.data_ptr());
      *dptr = value;
        */
}

/**
  | When filling a number to 1-element CPU tensor,
  | we want to skip everything but manipulate data
  | ptr directly.
  |
  | Ideally this fast pass should be implemented in
  | TensorIterator, but we also want to skip
  | compute_types which in not avoidable in
  | TensorIterator for now.
  |
  */
pub fn scalar_fill(
    self_: &mut Tensor,
    value: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          kHalf, kBool, kBFloat16, self.scalar_type(), "fill_out", [&]() {
            fill_inplace<Scalar>(self, value);
          });
      return self;
        */
}

pub fn scalar_tensor_static(
    s:          &Scalar,
    dtype_opt:  Option<ScalarType>,
    device_opt: Option<Device>) -> Tensor {
    
    todo!();
        /*
            tracer::NoTracerDispatchMode tracer_guard;
      AutoDispatchBelowAutograd mode;
      auto result = empty_cpu({}, dtype_opt, nullopt, device_opt, nullopt, nullopt);
      scalar_fill(result, s);
      return result;
        */
}
