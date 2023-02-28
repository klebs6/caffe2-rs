crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/PowKernel.cpp]

pub fn pow_tensor_tensor_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            const auto dtype = iter.common_dtype();
      if (isFloatingType(dtype) || isComplexType(dtype)) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, dtype, "pow", [&]() {

          using Vec = Vectorized<Scalar>;
          cpu_kernel_vec(iter,
            [=](Scalar base, Scalar exp) -> Scalar {
              return pow(base, exp);
            },
            [&](Vec base, Vec exp) -> Vec {
              return base.pow(exp);
            }
          );
        });
      } else {
        AT_DISPATCH_INTEGRAL_TYPES(dtype, "pow", [&]() {
          cpu_kernel(iter,
            [=](Scalar base, Scalar exp) -> Scalar {
              return native::powi(base, exp);
            }
          );
        });
      }
        */
}

/**
  | The source-code of kernels for float, double
  | and complex types is similar, barring a small
  | distinction - even if the output dtype is
  | float, a double exponent can be used. But
  | Complex types' computation doesn't allow
  | standard & double-precision to be mixed, since
  | pow takes either complex64 inputs, or
  | complex128 inputs, but not both.
  |
  | So, in order to provide a common path for
  | float, double & complex types, template
  | parameter cast_scalar_t is being used to
  | resolve the aforementioned distinction. This
  | approach also allows BFloat16 to use this
  | common-path.
  |
  | Half cannot currently use it, as AVX2 support
  | for sqrt & rsqrt doesn't currently exist for
  | it.
  */
pub fn pow_tensor_scalar_optimized_kernel<Scalar, cast_scalar_t, exp_scalar_t>(
        iter: &mut TensorIteratorBase,
        exp:  ExpScalar)  {

    todo!();
        /*
            using Vec = Vectorized<Scalar>;
      // .5 (sqrt), -.5 (rsqrt) and -1 (reciprocal) specializations are handled
      // in pow_tensor_scalar_kernel
      if (exp == 2.0) {
        cpu_kernel_vec(iter,
            [](Scalar base) -> Scalar {
              return base * base;
            },
            [](Vec base) -> Vec { return base * base; }
        );
      } else if (exp == 3.0) {
        cpu_kernel_vec(iter,
            [](Scalar base) -> Scalar {
              return base * base * base;
            },
            [](Vec base) -> Vec { return base * base * base; }
        );
      } else if (exp == -2.0) {
        cpu_kernel_vec(iter,
            [](Scalar base) -> Scalar {
              return static_cast<cast_scalar_t>(1.0) / (base * base); },
            [](Vec base) -> Vec { return (base * base).reciprocal(); }
        );
      } else {
        cpu_kernel_vec(iter,
            [=](Scalar base) -> Scalar {
              return pow(base, static_cast<cast_scalar_t>(exp));
            },
            [=](Vec base) -> Vec {
              return base.pow(static_cast<cast_scalar_t>(exp));
            }
        );
      }
        */
}

pub fn pow_tensor_scalar_kernel(
        iter:       &mut TensorIteratorBase,
        exp_scalar: &Scalar)  {
    
    todo!();
        /*
            // prevent multiple calls to iter.common_dtype()
      const auto dtype = iter.common_dtype();

      if (dtype == ScalarType::Float || dtype == ScalarType::Double ||
          dtype == kBFloat16 || isComplexType(dtype)) {
        // Dispatch to fast specialization for sqrt, rsqrt and reciprocal
        if (exp_scalar.equal(.5)) {
          return sqrt_kernel(iter);
        } else if (exp_scalar.equal(-0.5)) {
          return rsqrt_kernel(iter);
        } else if (exp_scalar.equal(-1.0)) {
          return reciprocal_kernel(iter);
        }
      }

      if (dtype == ScalarType::Float || dtype == ScalarType::Double) {
        AT_DISPATCH_FLOATING_TYPES(dtype, "pow", [&]() {
          pow_tensor_scalar_optimized_kernel<Scalar, double>(
              iter, exp_scalar.to<double>());
        });
      } else if (isComplexType(dtype)) {
        AT_DISPATCH_COMPLEX_TYPES(dtype, "pow", [&]() {
          pow_tensor_scalar_optimized_kernel<Scalar, Scalar>(
              iter, exp_scalar.to<complex<double>>());
        });
      } else if (dtype == ScalarType::Half) {
        [&]() {
          using Scalar =
              decltype(ScalarTypeToCPPType<ScalarType::Half>::t);
          const auto exp = exp_scalar.to<Scalar>();
          using Vec = Vectorized<Scalar>;
          cpu_kernel_vec(iter,
              [=](Scalar base) -> Scalar {
                return pow(base, exp);
              },
              [=](Vec base) -> Vec { return base.pow(exp); }
          );
        }();
      } else if (dtype == ScalarType::BFloat16) {
          AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, dtype, "pow", [&]() {
            pow_tensor_scalar_optimized_kernel<Scalar, Scalar>(
                iter, exp_scalar.to<Scalar>());
          });
      } else {
        AT_DISPATCH_INTEGRAL_TYPES(dtype, "pow", [&]() {
          const Scalar exp = exp_scalar.to<Scalar>();
          cpu_kernel(iter, [=](Scalar base) -> Scalar {
            return native::powi(base, exp);
          });
        });
      }
        */
}

register_dispatch!{pow_tensor_tensor_stub, &CPU_CAPABILITY::pow_tensor_tensor_kernel}
register_dispatch!{pow_tensor_scalar_stub, &CPU_CAPABILITY::pow_tensor_scalar_kernel}
