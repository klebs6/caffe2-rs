crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/ComplexKernel.cpp]

pub fn complex_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "complex_cpu", [&]() {
        cpu_kernel(iter, [=](Scalar a, Scalar b) -> complex<Scalar> {
          return complex<Scalar>(a, b);
        });
      });
        */
}

pub fn polar_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "polar_cpu", [&]() {
        cpu_kernel(iter, [=](Scalar a, Scalar b) -> complex<Scalar> {
          return complex<Scalar>(a * cos(b), a * sin(b));
        });
      });
        */
}

register_dispatch!{complex_stub , &complex_kernel}
register_dispatch!{polar_stub   , &polar_kernel}
