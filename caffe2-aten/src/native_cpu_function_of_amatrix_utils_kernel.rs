crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/FunctionOfAMatrixUtilsKernel.cpp]
pub fn compute_linear_combination_cpu_kernel(
        iter:           &mut TensorIterator,
        in_stride:      i64,
        coeff_stride:   i64,
        num_summations: i64)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
        iter.dtype(),
        "_compute_linear_combination_cpu", [&] {
          auto loop = [&](char** data, const i64* strides, i64 n) {
            auto* RESTRICT out_ptr = data[0];
            auto* RESTRICT in_ptr = data[1];
            auto* RESTRICT coeff_ptr = data[2];

            for (i64 elem = 0; elem < n; ++elem) {
              auto* RESTRICT out_data = reinterpret_cast<Scalar*>(out_ptr);
              auto* RESTRICT in_data = reinterpret_cast<Scalar*>(in_ptr);
              using primitive_t = typename scalar_value_type<Scalar>::type;
              auto* RESTRICT coeff_data = reinterpret_cast<primitive_t*>(coeff_ptr);

              // perform summation
              for (i32 i = 0; i < num_summations; ++i) {
                *out_data += in_data[i * in_stride] * coeff_data[i * coeff_stride];
              }

              out_ptr += strides[0];
              in_ptr += strides[1];
              coeff_ptr += strides[2];
            }
          };
          iter.for_each(loop);
      });
        */
}

register_dispatch!{_compute_linear_combination_stub, &_compute_linear_combination_cpu_kernel}
