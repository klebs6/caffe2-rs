crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp]

pub fn addr_kernel(
    iter:  &mut TensorIterator,
    beta:  &Scalar,
    alpha: &Scalar)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Bool) {
        using Scalar = bool;
        auto beta_val = beta.to<Scalar>();
        auto alpha_val = alpha.to<Scalar>();

        // when beta is false, values in self should be ignored,
        // nans and infs in self should not propagate.
        if (beta_val == false) {
          cpu_kernel(iter,
            [=](Scalar self_val,
                Scalar vec1_val,
                Scalar vec2_val) __ubsan_ignore_undefined__ -> Scalar {
              return alpha_val && vec1_val && vec2_val;
            }
          );
        } else {
          cpu_kernel(iter,
            [=](Scalar self_val,
                Scalar vec1_val,
                Scalar vec2_val) __ubsan_ignore_undefined__ -> Scalar {
              return (beta_val && self_val) || (alpha_val && vec1_val && vec2_val);
            }
          );
        }
        return;
      }

      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf,
        iter.dtype(), "addr_cpu", [&]() {
          using Vec = Vectorized<Scalar>;

          auto beta_val = beta.to<Scalar>();
          auto alpha_val = alpha.to<Scalar>();

          auto beta_vec = Vec(beta_val);
          auto alpha_vec = Vec(alpha_val);

          const Scalar zero_val(0);
          // when beta == 0, values in self should be ignored,
          // nans and infs in self should not propagate.
          if (beta_val == zero_val) {
            cpu_kernel_vec(iter,
              [=](Scalar self_val,
                  Scalar vec1_val,
                  Scalar vec2_val) __ubsan_ignore_undefined__ -> Scalar {
                return alpha_val * vec1_val * vec2_val;
              },
              [=](Vec self_vec,
                  Vec vec1_vec,
                  Vec vec2_vec) __ubsan_ignore_undefined__ {
                return alpha_vec * vec1_vec * vec2_vec;
              }
            );
          } else {
            cpu_kernel_vec(iter,
              [=](Scalar self_val,
                  Scalar vec1_val,
                  Scalar vec2_val) __ubsan_ignore_undefined__ -> Scalar {
                return beta_val * self_val + alpha_val * vec1_val * vec2_val;
              },
              [=](Vec self_vec,
                  Vec vec1_vec,
                  Vec vec2_vec) __ubsan_ignore_undefined__ {
                return beta_vec * self_vec + alpha_vec * vec1_vec * vec2_vec;
              }
            );
          }
        }
      );
        */
}

pub trait HasType {
    type Type;
}

pub fn linalg_vector_norm_kernel_cpu_impl<Scalar, Acc = <ScalarValueType<Scalar> as HasType>::Type>(
    iter: &mut TensorIterator,
    ord:  Scalar)  {

    todo!();
        /*
      double ord_val;
      if (ord.isFloatingPoint()) {
         ord_val = ord.to<double>();
      } else {
         TORCH_CHECK(false, "linalg.vector_norm expects ord to be float");
      }
      Acc init_val = (ord_val == -INFINITY) ? numeric_limits<Acc>::infinity() : static_cast<Acc>(0);
      if (iter.numel() == 0) {
        iter.output().fill_((ord_val < 0) ? INFINITY : 0);
        return;
      }
      if (ord_val == 0) {
        binary_kernel_reduce(iter, NormZeroOps<Scalar, Acc>(), init_val);
      } else if (ord_val == 1) {
        binary_kernel_reduce(iter, NormOneOps<Scalar, Acc>(), init_val);
      } else if (ord_val == 2) {
        binary_kernel_reduce(iter, NormTwoOps<Scalar, Acc>(), init_val);
      } else if (ord_val == INFINITY) {
        binary_kernel_reduce(iter, AbsMaxOps<Scalar, Acc>(), init_val);
      } else if (ord_val == -INFINITY) {
        binary_kernel_reduce(iter, AbsMinOps<Scalar, Acc>(), init_val);
      } else {
        binary_kernel_reduce(iter, NormOps<Scalar, Acc> { static_cast<Acc>(ord_val) }, init_val);
      }
      // For complex outputs, the above kernels do not touch the imaginary values,
      // so we must zero them out
      if (isComplexType(iter.output().scalar_type())) {
        imag(iter.output()).zero_();
      }
        */
}

pub fn linalg_vector_norm_kernel_cpu(
        iter: &mut TensorIterator,
        ord:  Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "linalg_vector_norm_cpu", [&] {
        linalg_vector_norm_kernel_cpu_impl<Scalar>(iter, ord);
      });
        */
}

pub fn unpack_pivots_cpu_kernel(
        iter:     &mut TensorIterator,
        dim_size: i64)  {
    
    todo!();
        /*
            if (iter.numel() == 0) {
        return;
      }

      auto loop = [&](char** data, const i64* strides, i64 nelems) {
        auto* unpacked_pivots_ptr = data[0];
        const auto* pivots_ptr = data[1];

        for (i64 elem = 0; elem < nelems; ++elem) {
          // WARNING: torch.lu returns int32 pivots,
          // this behavior could change in the future.
          auto* unpacked_pivots_data = reinterpret_cast<i32*>(unpacked_pivots_ptr);
          auto* pivots_data = reinterpret_cast<const i32*>(pivots_ptr);

          for (i64 i = 0; i < dim_size; ++i) {
            swap(
              unpacked_pivots_data[i],
              unpacked_pivots_data[pivots_data[i]]
            );
          }

          unpacked_pivots_ptr += strides[0];
          pivots_ptr += strides[1];
        }
      };

      iter.for_each(loop);
        */
}

register_dispatch!{addr_stub               , &addr_kernel}
register_dispatch!{linalg_vector_norm_stub , &linalg_vector_norm_kernel_cpu}
register_dispatch!{unpack_pivots_stub      , &unpack_pivots_cpu_kernel}
