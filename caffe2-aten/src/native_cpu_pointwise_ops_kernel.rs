/**
  | Ternary and higher-order pointwise
  | operations
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp]
pub fn addcmul_cpu_kernel(
        iter:  &mut TensorIterator,
        value: &Scalar)  {
    
    todo!();
        /*
            ScalarType dtype = iter.dtype(0);
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX(dtype, "addcmul_cpu_out", [&] {
        Scalar scalar_val = value.to<Scalar>();
        auto scalar_vec = Vectorized<Scalar>(scalar_val);
        cpu_kernel_vec(
            iter,
            [=](Scalar self_val, Scalar t1_val, Scalar t2_val) -> Scalar {
              return self_val + scalar_val * t1_val * t2_val;
            },
            [=](Vectorized<Scalar> self_vec,
                Vectorized<Scalar> t1_vec,
                Vectorized<Scalar> t2_vec) {
              return self_vec + scalar_vec * t1_vec * t2_vec;
            });
      });
        */
}

pub fn addcdiv_cpu_kernel(
        iter:  &mut TensorIterator,
        value: &Scalar)  {
    
    todo!();
        /*
            ScalarType dtype = iter.dtype(0);
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX(dtype, "addcdiv_cpu_out", [&] {
        Scalar scalar_val = value.to<Scalar>();
        auto scalar_vec = Vectorized<Scalar>(scalar_val);
        cpu_kernel_vec(
            iter,
            [=](Scalar self_val, Scalar t1_val, Scalar t2_val) -> Scalar {
              return self_val + scalar_val * t1_val / t2_val;
            },
            [=](Vectorized<Scalar> self_vec,
                Vectorized<Scalar> t1_vec,
                Vectorized<Scalar> t2_vec) {
              return self_vec + scalar_vec * t1_vec / t2_vec;
            });
      });
        */
}

pub fn smooth_l1_backward_cpu_kernel(
        iter: &mut TensorIterator,
        norm: &Scalar,
        beta: f64)  {
    
    todo!();
        /*
            ScalarType dtype = iter.dtype(0);
      AT_DISPATCH_ALL_TYPES(dtype, "smooth_l1_backward_cpu_out", [&] {
        auto norm_val = norm.to<Scalar>();
        Scalar beta_val(beta);
        auto norm_val_vec = Vectorized<Scalar>(norm_val);
        auto beta_val_vec = Vectorized<Scalar>(beta_val);
        const auto neg_1_vec = Vectorized<Scalar>(-1);
        const auto zero_vec = Vectorized<Scalar>(0);
        const auto pos_1_vec = Vectorized<Scalar>(1);
        cpu_kernel_vec(iter,
          [=](Scalar input, Scalar target, Scalar grad_output) -> Scalar {
            const auto x = input - target;
            if (x <= -beta)
              return -norm_val * grad_output;
            else if (x >= beta)
              return norm_val * grad_output;
            else
              return norm_val * x * grad_output / beta;
          },
          [norm_val_vec, beta_val_vec, neg_1_vec, zero_vec, pos_1_vec](
             Vectorized<Scalar> input, Vectorized<Scalar> target, Vectorized<Scalar> grad_output) -> Vectorized<Scalar> {
            // using two blendv calls to simulate the 3 cases
            // 1        if  x >= beta
            // -1       if x <= -beta
            // x / beta if |x| < beta
            const auto x = input - target;
            const auto pos_or_neg_1_vec = Vectorized<Scalar>::blendv(
                neg_1_vec, pos_1_vec, x > zero_vec);
            const auto x_abs = x.abs();
            const auto output = Vectorized<Scalar>::blendv(
                x / beta_val_vec, pos_or_neg_1_vec, x_abs >= beta_val_vec);
            return norm_val_vec * output * grad_output;
          }
        );
      });
        */
}

pub fn huber_backward_cpu_kernel(
        iter:  &mut TensorIterator,
        norm:  &Scalar,
        delta: f64)  {
    
    todo!();
        /*
            ScalarType dtype = iter.dtype(0);
      AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, dtype, "huber_backward_cpu_out", [&] {
        auto norm_val = norm.to<Scalar>();
        Scalar delta_val(delta);
        auto norm_val_vec = Vectorized<Scalar>(norm_val);
        auto delta_val_vec = Vectorized<Scalar>(delta_val);
        const auto neg_1_vec = Vectorized<Scalar>(-1);
        const auto zero_vec = Vectorized<Scalar>(0);
        const auto pos_1_vec = Vectorized<Scalar>(1);
        cpu_kernel_vec(iter,
          [=](Scalar input, Scalar target, Scalar grad_output) -> Scalar {
            const auto x = input - target;
            if (x <= -delta) {
              return -norm_val * grad_output * delta;
            } else if (x >= delta) {
              return norm_val * grad_output * delta;
            } else {
              return norm_val * x * grad_output;
            }
          },
          [norm_val_vec, delta_val_vec, neg_1_vec, zero_vec, pos_1_vec](
             Vectorized<Scalar> input, Vectorized<Scalar> target, Vectorized<Scalar> grad_output) -> Vectorized<Scalar> {
            // using two blendv calls to simulate the 3 cases
            // delta     if  x >= delta
            // -delta    if x <= -delta
            // x        if |x| < delta
            const auto x = input - target;
            const auto pos_or_neg_1_vec = Vectorized<Scalar>::blendv(
                neg_1_vec, pos_1_vec, x > zero_vec);
            const auto x_abs = x.abs();
            const auto output = Vectorized<Scalar>::blendv(
                x, pos_or_neg_1_vec * delta_val_vec, x_abs >= delta_val_vec);
            return norm_val_vec * output * grad_output;
          }
        );
      });
        */
}

pub fn mse_backward_cpu_kernel(
        iter:  &mut TensorIterator,
        value: &Scalar)  {
    
    todo!();
        /*
            ScalarType dtype = iter.dtype(0);
      AT_DISPATCH_ALL_TYPES(dtype, "mse_backward_cpu_out", [&] {
        Scalar scalar_val = value.to<Scalar>();
        auto scalar_vec = Vectorized<Scalar>(scalar_val);
        cpu_kernel_vec(
            iter,
            [=](Scalar self_val, Scalar t1_val, Scalar t2_val) -> Scalar {
              return scalar_val * (self_val - t1_val) * t2_val;
            },
            [=](Vectorized<Scalar> self_vec,
                Vectorized<Scalar> t1_vec,
                Vectorized<Scalar> t2_vec) {
              return scalar_vec * (self_vec - t1_vec) *  t2_vec;
        });
      });
        */
}

register_dispatch!{addcmul_stub            , &addcmul_cpu_kernel}
register_dispatch!{addcdiv_stub            , &addcdiv_cpu_kernel}
register_dispatch!{smooth_l1_backward_stub , &smooth_l1_backward_cpu_kernel}
register_dispatch!{huber_backward_stub     , &huber_backward_cpu_kernel}
register_dispatch!{mse_backward_stub       , &mse_backward_cpu_kernel}
