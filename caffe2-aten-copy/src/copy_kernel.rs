crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/CopyKernel.cpp]

pub fn copy_kernel(
        iter:         &mut TensorIterator,
        non_blocking: bool)  {
    
    todo!();
        /*
            ScalarType dtype = iter.dtype(0);
      if (dtype == iter.dtype(1)) {
        // TODO: as the majority of these operations can be done treating
        // their datatypes as opaque bit patterns, we don't actually need
        // separate instantiations per dtype; we only need a separate
        // instantiation per dtype size.  This would probably save us a
        // little bit of code size here
        // TODO: not sure if optimizer is able to compile two levels of
        // conditionals into a single jump table.  We should have a
        // single jump table here; might be worth just writing out the
        // dispatch statement by hand instead of using AT_DISPATCH
        if (dtype == ScalarType::Half) {
          cpu_kernel(iter, [=](Half a) -> Half { return a; });
        } else if (dtype == ScalarType::ComplexHalf) {
          cpu_kernel(iter, [=](complex<Half> a) -> complex<Half> { return a; });
        } else if (isQIntType(dtype)) {
          AT_DISPATCH_QINT_TYPES(dtype, "copy_kernel", [&] {
            cpu_kernel_vec(
                iter,
                [=](Scalar a) -> Scalar { return a; },
                [=](Vectorized<Scalar> a) -> Vectorized<Scalar> { return a; });
          });
        } else if (isComplexType(dtype)) {
          if (iter.tensor(0).is_conj() == iter.tensor(1).is_conj()) {
            AT_DISPATCH_COMPLEX_TYPES(dtype, "copy_kernel", [&] {
                cpu_kernel_vec(
                  iter,
                  [=](Scalar a) -> Scalar { return a; },
                  [=](Vectorized<Scalar> a) -> Vectorized<Scalar> { return a; });
              });
          } else {
            AT_DISPATCH_COMPLEX_TYPES(dtype, "conj_kernel", [&] {
                cpu_kernel_vec(
                  iter,
                  [=](Scalar a) -> Scalar { return conj_impl(a); },
                  [=](Vectorized<Scalar> a) -> Vectorized<Scalar> { return a.conj(); });
              });
          }
        } else {
          AT_DISPATCH_ALL_TYPES_AND2(
              ScalarType::Bool, ScalarType::BFloat16,dtype, "copy_kernel", [&] {
                cpu_kernel_vec(
                    iter,
                    [=](Scalar a) -> Scalar { return a; },
                    [=](Vectorized<Scalar> a) { return a; });
              });
        }
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, dtype, "copy_", [&] {
          using dest_t = Scalar;
          AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, iter.dtype(1), "copy_", [&] {
            // Note (@zasdfgbnm):
            //
            // The code below can not be simplified as
            //    cpu_kernel(iter, static_cast_with_inter_type<dest_t, Scalar>::apply);
            //
            // because this would force the compiler to instantiate the inline function and generate a function call in the loop
            // instead of inlining it, making all the optimizations like vectorization impossible.
            // You can verify this by looking the the symbols of `libtorch_cpu.so`:
            //
            //    readelf -Ws libtorch_cpu.so | grep static_cast_with_inter_type
            //
            // If done correctly, the above command should have no output.
            //
            // See: https://github.com/pytorch/pytorch/issues/31271
            cpu_kernel(iter, [](Scalar src) -> dest_t {
              return static_cast_with_inter_type<dest_t, Scalar>::apply(src); });
          });
        });
        if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
          iter.tensor(0).conj_physical_();
        }
      }
        */
}

register_dispatch!{copy_stub, &copy_kernel}
