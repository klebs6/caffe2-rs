crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/FillKernel.cpp]

pub fn fill_non_native_type<Scalar>(
    iter:         &mut TensorIterator,
    value_scalar: &Scalar)  {

    todo!();
        /*
            auto value = value_scalar.to<Scalar>().x;
      using H = typename make_signed<decltype(value)>::type;  // Signed type has more acceleration
      // Reserve the representation of value. static_cast<H>(value) is implementation defined.
      H val = *reinterpret_cast<H*>(addressof(value));
      cpu_kernel_vec</*check_dynamic_cast=*/false>(
          iter,
          [val]() -> H { return val; },
          [val]() { return Vectorized<H>(val); });
        */
}

pub fn fill_non_native_type_complex_half(
        iter:         &mut TensorIterator,
        value_scalar: &Scalar)  {
    
    todo!();
        /*
            static_assert(sizeof(complex<Half>) == sizeof(i32), "Size of ComplexHalf should be 32-bits");
      auto value = complex<Half>(value_scalar.to<complex<float>>());
      auto val = *reinterpret_cast<i32*>(addressof(value));
      cpu_kernel_vec</*check_dynamic_cast=*/false>(
          iter,
          [val]() -> i32 { return val; },
          [val]() { return Vectorized<i32>(val); });
        */
}

pub fn fill_kernel(
        iter:         &mut TensorIterator,
        value_scalar: &Scalar)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Half) {
        fill_non_native_type<Half>(iter, value_scalar);
      } else if (iter.dtype() == ScalarType::BFloat16) {
        fill_non_native_type<BFloat16>(iter, value_scalar);
      } else if (iter.dtype() == ScalarType::ComplexHalf) {
        fill_non_native_type<complex<Half>>(iter, value_scalar);
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Bool, iter.dtype(), "fill_cpu", [&]() {
          Scalar value = value_scalar.to<Scalar>();
          cpu_kernel_vec(
              iter,
              [=]() -> Scalar { return value; },
              [=]() { return Vectorized<Scalar>(value); });
        });
      }
        */
}

register_dispatch!{fill_stub, &fill_kernel}
