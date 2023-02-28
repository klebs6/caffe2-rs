crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp]

#[inline] pub fn cpu_cum_base_kernel<Scalar, func_t>(
    result:   &mut Tensor,
    self_:    &Tensor,
    dim:      i64,
    f:        &Func,
    init_val: Scalar)  {

    todo!();
        /*
            if (result.sizes() != self.sizes()) {
        at::native::resize_output(result, self.sizes());
      }
      if (self.numel() == 0) {
        return;
      }
      const auto input_ndim = self.dim();
      if (input_ndim == 0) {
        result.fill_(self);
        return;
      }

      // TODO This probably should be using at::native::make_reduction
      auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .declare_static_shape(self.sizes(), /*squash_dim=*/dim)
        .add_output(result)
        .add_input(self)
        .build();

      auto result_dim_stride = ensure_nonempty_stride(result, dim);
      auto self_dim_stride = ensure_nonempty_stride(self, dim);

      auto loop = [&](char** data, const i64* strides, i64 n) {
        auto* result_data_bytes = data[0];
        const auto* self_data_bytes = data[1];

        for (i64 i = 0; i < n; ++i) {
          f(
            (Scalar*)result_data_bytes, result_dim_stride,
            (Scalar*)self_data_bytes, self_dim_stride, init_val
          );
          result_data_bytes += strides[0];
          self_data_bytes += strides[1];
        }
      };

      iter.for_each(loop);
        */
}

pub fn cumsum_cpu_kernel(
    result: &mut Tensor,
    self_:  &Tensor,
    dim:    i64)  {

    todo!();
        /*
            auto wrap_dim = maybe_wrap_dim(dim, self.dim());
      i64 self_dim_size = ensure_nonempty_size(self, wrap_dim);

      AT_DISPATCH_ALL_TYPES_AND_COMPLEX(self.scalar_type(), "cumsum_out_cpu", [&] {
        cpu_cum_base_kernel<Scalar>(result, self, wrap_dim, [&] (
          Scalar* result_data, auto result_dim_stride,
          const Scalar* self_data, auto self_dim_stride, Scalar init_val) {
            auto cum_number = (at::acc_type<Scalar, false>)init_val;
            for (i64 i = 0; i < self_dim_size; ++i) {
              cum_number += self_data[i * self_dim_stride];
              result_data[i * result_dim_stride] = (Scalar)cum_number;
            }
          }, /*init_val=*/ 0
        );
      });
        */
}

pub fn cumprod_cpu_kernel(
    result: &mut Tensor,
    self_:  &Tensor,
    dim:    i64)  {

    todo!();
        /*
            auto wrap_dim = maybe_wrap_dim(dim, self.dim());
      i64 self_dim_size = ensure_nonempty_size(self, wrap_dim);

      AT_DISPATCH_ALL_TYPES_AND_COMPLEX(self.scalar_type(), "cumprod_out_cpu", [&] {
        cpu_cum_base_kernel<Scalar>(result, self, wrap_dim, [&] (
          Scalar* result_data, auto result_dim_stride,
          const Scalar* self_data, auto self_dim_stride, Scalar init_val) {
            auto cum_number = (at::acc_type<Scalar, false>)init_val;
            for (i64 i = 0; i < self_dim_size; ++i) {
              cum_number *= self_data[i * self_dim_stride];
              result_data[i * result_dim_stride] = (Scalar)cum_number;
            }
          }, /*init_val=*/ 1
        );
      });
        */
}

pub fn logcumsumexp_cpu_kernel(
    result: &mut Tensor,
    self_:  &Tensor,
    dim:    i64)  {
    
    todo!();
        /*
            auto wrap_dim = maybe_wrap_dim(dim, self.dim());
      i64 self_dim_size = ensure_nonempty_size(self, wrap_dim);

      AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "logcumsumexp_out_cpu", [&] {
        cpu_cum_base_kernel<Scalar>(result, self, wrap_dim, [&] (
          Scalar* result_data, auto result_dim_stride,
          const Scalar* self_data, auto self_dim_stride, Scalar init_val) {
            Scalar cum_number = (at::acc_type<Scalar, false>)init_val;
            for (i64 i = 0; i < self_dim_size; ++i) {
              Scalar x = self_data[i * self_dim_stride];

              // Reference : https://www.tensorflow.org/api_docs/python/tf/math/cumulative_logsumexp
              auto log_add_exp = [](Scalar x, Scalar y) -> Scalar {
                Scalar min = std::isnan(y) ? y : std::min(x,y); //std::min returns first arg if one of the args is nan
                Scalar max = std::isnan(y) ? y : std::max(x,y); //std::max returns first arg if one of the args is nan
                if (min != max || std::isfinite(min)) {
                  // nan will be propagated here
                  return std::log1p(std::exp(min - max)) + max;
                } else {
               // special case to correctly handle infinite cases
                  return x;
                }
              };
              cum_number = log_add_exp(x, cum_number);
              result_data[i * result_dim_stride] = static_cast<Scalar>(cum_number);
            }
          }, /*init_val=*/ -std::numeric_limits<Scalar>::infinity()
        );
      });
        */
}

/**
  | TODO: Implement `nansum` similar to
  | the stable `sum` implementation in
  | cpu/SumKernel.cpp
  |
  */
pub fn nansum_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Half){
        binary_kernel_reduce(iter, NanSumOps<float, c10::Half>{}, float{0});
      } else {
        AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "nansum_cpu", [&] {
        binary_kernel_reduce(iter, NanSumOps<Scalar, Scalar>{}, Scalar{0});
      });
      }
        */
}

pub fn mean_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "mean_cpu", [&] {
        Scalar factor = Scalar(iter.num_output_elements()) / Scalar(iter.numel());
        binary_kernel_reduce(
          iter,
          MeanOps<Scalar, Scalar> {factor},
          Scalar(0)
        );
      });
        */
}

pub fn std_var_kernel_impl(
    iter:       &mut TensorIterator,
    correction: i64,
    take_sqrt:  bool)  {

    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "std_cpu", [&] {
        binary_kernel_reduce(
            iter,
            WelfordOps<
                Scalar,
                double,
                i64,
                double,
                std::tuple<Scalar, Scalar>>{correction, take_sqrt},
            WelfordData<double, i64, double>());
      });
        */
}

pub fn prod_kernel_impl(iter: &mut TensorIterator)  {

    todo!();
        /*
            // Workaround for the error: '*' in boolean context, suggest '&&' instead [-Werror=int-in-bool-context]
      if (iter.dtype() == ScalarType::Bool) {
        using Scalar = bool;
        binary_kernel_reduce_vec(
          iter,
          [=](Scalar a, Scalar b) -> Scalar { return a && b; },
          [=](Vectorized<Scalar> a, Vectorized<Scalar> b) { return a && b; },
          /*identity=*/1);
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "prod_cpu", [&] {
          binary_kernel_reduce_vec(
            iter,
            [=](Scalar a, Scalar b) -> Scalar { return a * b; },
            [=](Vectorized <Scalar> a, Vectorized <Scalar> b) { return a * b; },
            /*identity=*/1);
          });
      }
        */
}

pub fn norm_kernel_tensor_iterator_impl(
    iter: &mut TensorIterator,
    p:    &Scalar)  {
    
    todo!();
        /*
      float val;
      if (p.isIntegral(false)) {
        val = p.to<i64>();
      } else if (p.isFloatingPoint()) {
        val = p.to<double>();
      } else {
        AT_ERROR("norm_kernel_tensor_iterator_impl expects norm to be integer or float");
      }

      // In the dispatch code blocks below, reduction kernels accumulate results as
      // the type `Acc`. When `Scalar` is complex, `Acc` is the downgraded
      // real number type. Otherwise, `Acc` and `Scalar` are the same type.
      if (val == 0) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
          using Acc = typename scalar_value_type<Scalar>::type;
          binary_kernel_reduce(
            iter,
            NormZeroOps<Scalar, Acc>(),
            Acc(0)
          );
        });
      } else if (val == 1) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
          using Acc = typename scalar_value_type<Scalar>::type;
          binary_kernel_reduce(
            iter,
            NormOneOps<Scalar, Acc>(),
            Acc(0)
          );
        });
      } else if (val == 2) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
          using Acc = typename scalar_value_type<Scalar>::type;
          binary_kernel_reduce(
            iter,
            NormTwoOps<Scalar, Acc>(),
            Acc(0)
          );
        });
      } else if (val == INFINITY) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
          using Acc = typename scalar_value_type<Scalar>::type;
          binary_kernel_reduce(
            iter,
            AbsMaxOps<Scalar, Acc>(),
            Acc(0)
          );
        });
      } else if (val == -INFINITY) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
          using Acc = typename scalar_value_type<Scalar>::type;
          binary_kernel_reduce(
            iter,
            AbsMinOps<Scalar, Acc>(),
            Acc::max
          );
        });
      } else {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
          using Acc = typename scalar_value_type<Scalar>::type;
          binary_kernel_reduce(
            iter,
            NormOps<Scalar, Acc> { Acc(val) },
            Acc(0)
          );
        });
      }

      // For complex outputs, the above kernels do not touch the imaginary values,
      // so we must zero them out
      if (isComplexType(iter.output().scalar_type())) {
        at::imag(iter.output()).zero_();
      }
        */
}

pub fn and_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Byte) {
        // Refer [all, any : uint8 compatibility]
        binary_kernel_reduce_vec(
            iter,
            [=](u8 a, u8 b) -> u8 { return (a && b) ? 1 : 0; },
            [=](Vectorized<u8> a, Vectorized<u8> b) {
              Vectorized<u8> c = Vectorized<u8>();

              for (decltype(c.size()) i = 0; i != Vectorized<u8>::size(); i++) {
                c[i] = (a[i] && b[i]) ? 1 : 0;
              }
              return c;
            },
            /*ident=*/true);
      } else {
        binary_kernel_reduce_vec(
            iter,
            [=](bool a, bool b) -> bool { return a && b; },
            [=](Vectorized<bool> a, Vectorized<bool> b) {
              // Adding the implementation here instead of in vec256_base to avoid
              // return value inconsistency. Other comparison operators in
              // vec256_base return -1/0 (all bit 1 / all bit 0) as true/false to
              // follow the AVX2 convention. This would be convenient when combined
              // with other vectorized operations. For example, one can use the
              // logical operation results as a mask for a bit operation to
              // retrieve/reset multiple elements in a vector.
              //
              // In this method, users would expect, e.g., all(), to return 1/0 as
              // true/false.
              Vectorized<bool> c = Vectorized<bool>();

              for (decltype(c.size()) i = 0; i != Vectorized<bool>::size(); i++) {
                c[i] = a[i] && b[i];
              }
              return c;
            },
            /*ident=*/true);
      }
        */
}

pub fn or_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Byte) {
        // Refer [all, any : uint8 compatibility]
        binary_kernel_reduce_vec(
            iter,
            [=](u8 a, u8 b) -> u8 { return (a || b) ? 1 : 0; },
            [=](Vectorized<u8> a, Vectorized<u8> b) {
              Vectorized<u8> c = Vectorized<u8>();

              for (decltype(c.size()) i = 0; i != Vectorized<u8>::size(); i++) {
                c[i] = (a[i] || b[i]) ? 1 : 0;
              }
              return c;
            },
            /*ident=*/false);
      } else {
        binary_kernel_reduce_vec(
            iter,
            [=](bool a, bool b) -> bool { return a || b; },
            [=](Vectorized<bool> a, Vectorized<bool> b) {
              Vectorized<bool> c = Vectorized<bool>();

              for (decltype(c.size()) i = 0; i != Vectorized<bool>::size(); i++) {
                c[i] = a[i] || b[i];
              }
              return c;
            },
            /*ident=*/false);
      }
        */
}

lazy_static!{
    /*
    template<typename Scalar>
    struct MinValuesOps: public at::native::MinOps<Scalar> {
      using arg_t = typename MinOps<Scalar>::arg_t;
      static Scalar project(arg_t arg) {
        return arg.first;
      }
    };
    */
}

pub fn min_values_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            if (iter.dtype() == kLong) {
        // This case is special because of Vectorized<i64> does not
        // handle upper_bound<i64>().
        // See: https://github.com/pytorch/pytorch/issues/43254
        using Scalar = i64;
        binary_kernel_reduce(
          iter,
          MinValuesOps<Scalar>{},
          std::pair<Scalar, i64>(upper_bound<Scalar>(), -1));
        return;
      }
      AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "min_values_cpu", [&iter] {
        binary_kernel_reduce_vec(
          iter,
          [](Scalar a, Scalar b) -> Scalar { return min_impl(a, b); },
          [](Vectorized<Scalar> a, Vectorized<Scalar> b) { return minimum(a, b); },
          static_cast<double>(upper_bound<Scalar>()));
      });
        */
}

pub fn max_values_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "max_values_cpu", [&iter] {
        binary_kernel_reduce_vec(
          iter,
          [](Scalar a, Scalar b) -> Scalar { return max_impl(a, b); },
          [](Vectorized<Scalar> a, Vectorized<Scalar> b) { return maximum(a, b); },
          lower_bound<Scalar>());
      });
        */
}

pub fn argmax_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(1), "argmax_cpu", [&] {
        binary_kernel_reduce(
          iter,
          ArgMaxOps<Scalar>{},
          std::pair<Scalar, i64>(lower_bound<Scalar>(), 0));
      });
        */
}

pub fn argmin_kernel_impl(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(1), "argmin_cpu", [&] {
        binary_kernel_reduce(
          iter,
          ArgMinOps<Scalar>{},
          std::pair<Scalar, i64>(upper_bound<Scalar>(), 0));
      });
        */
}

register_dispatch!{
    nansum_stub, 
    &nansum_kernel_impl
}

register_dispatch!{
    std_var_stub, 
    &std_var_kernel_impl
}

register_dispatch!{
    prod_stub, 
    &prod_kernel_impl
}

register_dispatch!{
    mean_stub, 
    &mean_kernel_impl
}

register_dispatch!{
    norm_stub, 
    &norm_kernel_tensor_iterator_impl
}

register_dispatch!{
    and_stub, 
    &and_kernel_impl
}

register_dispatch!{
    or_stub, 
    &or_kernel_impl
}

register_dispatch!{
    min_values_stub, 
    &min_values_kernel_impl
}

register_dispatch!{
    max_values_stub, 
    &max_values_kernel_impl
}

register_dispatch!{
    argmax_stub, 
    &argmax_kernel_impl
}

register_dispatch!{
    argmin_stub, 
    &argmin_kernel_impl
}

register_dispatch!{
    cumprod_stub, 
    &cumprod_cpu_kernel
}

register_dispatch!{
    cumsum_stub, 
    &cumsum_cpu_kernel
}

register_dispatch!{
    logcumsumexp_stub, 
    &logcumsumexp_cpu_kernel
}
