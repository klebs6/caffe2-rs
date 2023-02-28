crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/Activation.cpp]

#[inline] pub fn vec_log_sigmoid<Scalar>(
    output: &mut Tensor,
    buffer: &mut Tensor,
    input:  &Tensor)  {

    todo!();
        /*
            using Vec = Vectorized<Scalar>;
      Scalar* output_data = output.data_ptr<Scalar>();
      Scalar* buffer_data = buffer.data_ptr<Scalar>();
      Scalar* input_data = input.data_ptr<Scalar>();
      parallel_for(0, input.numel(), 1, [&] (i64 begin, i64 end) {
        i64 size = end - begin;
        i64 d = 0;
        for (; d < size - (size % Vec::size()); d += Vec::size()) {
          Vec data_vec = Vec::loadu(input_data + begin+ d);
          Vec max_vec = vec::maximum(data_vec.neg(), Vec(Scalar(0)));
          Vec buffer_vec =  max_vec.neg().exp() + (data_vec.neg() - max_vec).exp();
          Vec output_vec = (max_vec + buffer_vec.log()).neg();
          buffer_vec.store(buffer_data + begin + d);
          output_vec.store(output_data + begin + d);
        }
        if (size - d > 0) {
          Vec data_vec = Vec::loadu(input_data + begin + d, size - d);
          Vec max_vec = vec::maximum(data_vec.neg(), Vec(Scalar(0)));
          Vec buffer_vec =  max_vec.neg().exp() + (data_vec.neg() - max_vec).exp();
          Vec output_vec = (max_vec + buffer_vec.log()).neg();
          buffer_vec.store(buffer_data + begin + d, size - d);
          output_vec.store(output_data + begin + d, size - d);
        }
      });
        */
}

pub fn log_sigmoid_cpu_kernel(
    output: &mut Tensor,
    buffer: &mut Tensor,
    input:  &Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_sigmoid_cpu", [&] {
        _vec_log_sigmoid<Scalar>(output, buffer, input);
      });
        */
}

pub fn log_sigmoid_backward_cpu_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "log_sigmoid_backward_cpu", [&]() {
        using Vec = Vectorized<Scalar>;
        auto zero_val = Scalar(0);
        auto zero_vec = Vec(zero_val);
        auto one_val = Scalar(1);
        auto one_vec = Vec(one_val);
        cpu_kernel_vec(iter,
          [=](Scalar a, Scalar b, Scalar c) -> Scalar {
            auto max_deriv_val = zero_val;
            auto sign_val = -one_val;
            if (a < zero_val) {
              max_deriv_val = -one_val;
              sign_val = one_val;
            }
            return (-max_deriv_val - sign_val * ((b - one_val) / b)) * c;
          },
          [=](Vec a, Vec b, Vec c) -> Vec {
            auto mask = a < zero_vec;
            auto max_deriv_vec = Vec::blendv(zero_vec, one_vec.neg(), mask);
            auto sign_vec = Vec::blendv(one_vec.neg(), one_vec, mask);
            return (max_deriv_vec + sign_vec * ((b - one_vec) / b)).neg() * c;
          });
      });
        */
}

pub fn threshold_kernel(
    iter:             &mut TensorIteratorBase,
    threshold_scalar: &Scalar,
    value_scalar:     &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "threshold_cpu", [&] {
        using Vec = Vectorized<Scalar>;
        Scalar threshold = threshold_scalar.to<Scalar>();
        Vec threshold_v = Vec(threshold);
        Scalar value = value_scalar.to<Scalar>();
        Vec value_v = Vec(value);
        cpu_kernel_vec(
            iter,
            [&](Scalar x, Scalar other) -> Scalar {
              return x <= threshold ? value : other;
            },
            [&](Vec x, Vec other) -> Vec {
              return Vec::blendv(other, value_v, x <= threshold_v);
            });
      });
        */
}

#[cfg(AT_MKL_ENABLED)]
lazy_static!{
    /*
    template <typename T>
    void MKLCdfNorm(i64 N, const T* X, T* Y);

    template <>
    void MKLCdfNorm<float>(i64 N, const float* X, float* Y) {
      vsCdfNorm(N, X, Y);
    }

    template <>
    void MKLCdfNorm<double>(i64 N, const double* X, double* Y) {
      vdCdfNorm(N, X, Y);
    }

    template <typename T>
    void MKLMul(i64 N, const T* A, const T* B, T* Y);

    template <>
    void MKLMul<float>(i64 N, const float* A, const float* B, float* Y) {
      vsMul(N, A, B, Y);
    }

    template <>
    void MKLMul<double>(i64 N, const double* A, const double* B, double* Y) {
      vdMul(N, A, B, Y);
    }

    template <typename T>
    void MKLExp(i64 N, const T* X, T* Y);

    template <>
    void MKLExp<float>(i64 N, const float* X, float* Y) {
      vsExp(N, X, Y);
    }

    template <>
    void MKLExp<double>(i64 N, const double* X, double* Y) {
      vdExp(N, X, Y);
    }
    */
}


#[cfg(AT_MKL_ENABLED)]
pub fn gelu_mkl_kernel_impl<T>(it: *mut TensorIteratorBase)  {

    todo!();
        /*
            if (!it->can_use_32bit_indexing()) {
        for (auto& sub_it : it->with_32bit_indexing()) {
          GeluMKLKernelImpl<T>(&sub_it);
        }
        return;
      }
      const i64 N = it->numel();
      const T* X_data = static_cast<T*>(it->data_ptr(1));
      T* Y_data = static_cast<T*>(it->data_ptr(0));
      MKLCdfNorm<T>(N, X_data, Y_data);
      MKLMul<T>(N, X_data, Y_data, Y_data);
        */
}

#[cfg(AT_MKL_ENABLED)]
pub fn gelu_backward_mkl_kernel_impl<T>(it: *mut TensorIteratorBase)  {

    todo!();
        /*
            if (!it->can_use_32bit_indexing()) {
        for (auto& sub_it : it->with_32bit_indexing()) {
          GeluBackwardMKLKernelImpl<T>(&sub_it);
        }
        return;
      }
      constexpr T kBeta = M_2_SQRTPI * M_SQRT1_2 * T(0.5);
      const i64 N = it->numel();
      const T* dY_data = static_cast<T*>(it->data_ptr(1));
      const T* X_data = static_cast<T*>(it->data_ptr(2));
      T* dX_data = static_cast<T*>(it->data_ptr(0));
      Tensor cdf = at::empty({N}, it->input(1).options());
      T* cdf_data = cdf.template data_ptr<T>();
      MKLCdfNorm<T>(N, X_data, cdf_data);
      for (i64 i = 0; i < N; ++i) {
        dX_data[i] = T(-0.5) * X_data[i] * X_data[i];
      }
      MKLExp(N, dX_data, dX_data);
      for (i64 i = 0; i < N; ++i) {
        dX_data[i] = dY_data[i] * (cdf_data[i] + kBeta * X_data[i] * dX_data[i]);
      }
        */
}

#[cfg(not(AT_MKL_ENABLED))]
pub fn gelu_mkl_kernel_impl<T>(it: *mut TensorIteratorBase)  {

    todo!();
        /*
            TORCH_CHECK(false, "ATen not compiled with MKL");
        */
}

#[cfg(not(AT_MKL_ENABLED))]
pub fn gelu_backward_mkl_kernel_impl<T>(it: *mut TensorIteratorBase)  {

    todo!();
        /*
            TORCH_CHECK(false, "ATen not compiled with MKL");
        */
}

pub fn elu_kernel(
        it:          &mut TensorIteratorBase,
        alpha:       &Scalar,
        scale:       &Scalar,
        input_scale: &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(it.dtype(), "elu_cpu", [&]() {
        using Vec = Vectorized<Scalar>;
        auto negcoef = alpha.to<Scalar>() * scale.to<Scalar>();
        auto poscoef = scale.to<Scalar>();
        auto negiptcoef = input_scale.to<Scalar>();
        const Vec negcoef_vec(negcoef);
        const Vec negiptcoef_vec(negiptcoef);
        const Vec poscoef_vec(poscoef);
        const Vec one_vec(static_cast<Scalar>(1));
        const Vec zero_vec(static_cast<Scalar>(0));
        cpu_kernel_vec(
            it,
            [negcoef, negiptcoef, poscoef](Scalar a) -> Scalar {
              return a <= Scalar(0) ? (std::exp(a * negiptcoef) - Scalar(1)) * negcoef : a * poscoef;
            },
            [&negcoef_vec, &negiptcoef_vec, &poscoef_vec, &one_vec, &zero_vec](Vec a) -> Vec {
              auto cmp = (a > zero_vec);
              if (!cmp.zero_mask()) {  // only a * poscoef (which is very quick) needs to be computed
                return a * poscoef_vec;
              } else {
                return Vec::blendv(((a * negiptcoef_vec).exp() - one_vec) * negcoef_vec, a * poscoef_vec, cmp);
              }
            });
      });
        */
}

pub fn elu_backward_kernel(
        it:          &mut TensorIteratorBase,
        alpha:       &Scalar,
        scale:       &Scalar,
        input_scale: &Scalar,
        is_result:   bool)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(it.dtype(), "elu_backward_cpu", [&]() {
        using Vec = Vectorized<Scalar>;
        auto negcoef = alpha.to<Scalar>() * scale.to<Scalar>();
        auto poscoef = scale.to<Scalar>();
        auto negiptcoef = input_scale.to<Scalar>();
        const Vec negcoef_vec(negcoef);
        const Vec negiptcoef_vec(negiptcoef);
        const Vec poscoef_vec(poscoef);
        const Vec zero_vec(static_cast<Scalar>(0));
        cpu_kernel_vec(
            it,
            [negcoef, negiptcoef, poscoef, is_result](Scalar a, Scalar b) -> Scalar {
              if (is_result) {
                return b <= Scalar(0) ? a * negiptcoef * (b + negcoef) : a * poscoef;
              } else {
                return b <= Scalar(0) ? a * negiptcoef * negcoef * std::exp(b * negiptcoef): a * poscoef;
              }
            },
            [&negcoef_vec, &negiptcoef_vec, &poscoef_vec, &zero_vec, is_result](Vec a, Vec b) -> Vec {
              auto cmp = (b > zero_vec);
              if (is_result) {
                if (!cmp.zero_mask()) {  // only a * poscoef (which is very quick) needs to be computed
                  return a * poscoef_vec;
                } else {
                  return Vec::blendv(a * negiptcoef_vec * (b + negcoef_vec), a * poscoef_vec, cmp);
                }
              } else {
                return Vec::blendv(a * negiptcoef_vec * negcoef_vec * (b * negiptcoef_vec).exp(), a * poscoef_vec, cmp);
              }
            }
        );
      });
        */
}

/**
 | TODO(yangxm): Add another fast kernel using formula
 | y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
 | and the fast tanh impl from Eigen.
 */
pub fn gelu_kernel_impl(it: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            if (at::hasMKL() && it.is_contiguous()) {
        AT_DISPATCH_FLOATING_TYPES(it.dtype(), "GeluKernelImpl", [&]() {
          GeluMKLKernelImpl<Scalar>(&it);
        });
      } else {
        auto grain_size = at::internal::GRAIN_SIZE;
        // Numbers based on benchmarking.
        // Benchmark: benchmarks/operator_benchmarks/pt/gelu_test.py
    #ifdef C10_MOBILE
        // Benchmarked on S8 US phone.
        // Internal benchmarking that converts operator benchmark into
        // a torchscript module and run that on mobile.
        // Same benchmark as server side.
        constexpr i64 GELU_MIN_ELEMENTS_FOR_MULTI_THREADING{6144};
    #else
        // Benchmarked on i9 8 core 16 thread machine.
        // 1 thread: cd benchmark/operator_benchmarks;
        //           python -m pt.gelu_test --tag_filter long --omp_num_threads 1
        // 2 threads: cd benchmark/operator_benchmarks;
        //           python -m pt.gelu_test --tag_filter long --omp_num_threads 1
        constexpr i64 GELU_MIN_ELEMENTS_FOR_MULTI_THREADING{16384};
    #endif
        if (it.numel() > GELU_MIN_ELEMENTS_FOR_MULTI_THREADING) {
          grain_size = it.numel() / at::get_num_threads();
        }
        AT_DISPATCH_FLOATING_TYPES(it.dtype(), "GeluKernelImpl", [&]() {
          using Vec = vec::Vectorized<Scalar>;
          const Vec kAlphaVec(M_SQRT1_2);
          const Vec kOneVec(1);
          const Vec kPointFiveVec(0.5);
          cpu_kernel_vec(
              it,
              [](Scalar x) {
                constexpr Scalar kAlpha = M_SQRT1_2;
                return x * Scalar(0.5) * (Scalar(1) + std::erf(x * kAlpha));
              },
              [&](Vec x_vec) {
                return x_vec * kPointFiveVec *
                    (kOneVec + (x_vec * kAlphaVec).erf());
              },
              grain_size);
        });
      }
        */
}

pub fn gelu_backward_kernel_impl(it: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            if (hasMKL() && it.is_contiguous()) {
        AT_DISPATCH_FLOATING_TYPES(it.dtype(), "GeluBackwardKernelImpl", [&]() {
          GeluBackwardMKLKernelImpl<Scalar>(&it);
        });
      } else {
        AT_DISPATCH_FLOATING_TYPES(it.dtype(), "GeluBackwardKernelImpl", [&]() {
          using Vec = vec::Vectorized<Scalar>;
          const Vec kAlphaVec(M_SQRT1_2);
          const Vec kBetaVec(M_2_SQRTPI * M_SQRT1_2 * 0.5);
          const Vec kOneVec(1);
          const Vec kPointFiveVec(0.5);
          const Vec kMinusPointFiveVec(-0.5);
          cpu_kernel_vec(
              it,
              [](Scalar dy, Scalar x) {
                constexpr Scalar kAlpha = M_SQRT1_2;
                constexpr Scalar kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
                const Scalar cdf =
                    Scalar(0.5) * (Scalar(1) + std::erf(x * kAlpha));
                const Scalar pdf = kBeta * std::exp(x * x * Scalar(-0.5));
                return dy * (cdf + x * pdf);
              },
              [&](Vec dy_vec, Vec x_vec) {
                const Vec cdf_vec =
                    kPointFiveVec * (kOneVec + (x_vec * kAlphaVec).erf());
                const Vec pdf_vec =
                    kBetaVec * (x_vec * x_vec * kMinusPointFiveVec).exp();
                return dy_vec * (cdf_vec + x_vec * pdf_vec);
              });
        });
      }
        */
}

pub fn hardsigmoid_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardsigmoid_cpu", [&] {
        const Scalar zero(0.0f);
        const Scalar three(3.0f);
        const Scalar six(6.0f);
        using Vec = vec::Vectorized<Scalar>;
        const Vec kZeroVec(zero);
        const Vec kThreeVec(three);
        const Vec kSixVec(six);
        cpu_kernel_vec(
            iter,
            [&](Scalar self_val) {
              return std::min(std::max(self_val + three, zero), six) / six;
            },
            [&](Vec self_val) {
              return vec::minimum(
                vec::maximum(self_val + kThreeVec, kZeroVec),
                kSixVec
              ) / kSixVec;
            });
      });
        */
}

pub fn hardsigmoid_backward_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardsigmoid_backward", [&] {
        const Scalar zero(0.0f);
        const Scalar three(3.0f);
        const Scalar neg_three(-3.0f);
        const Scalar one_sixth(1.0f / 6.0f);
        using Vec = Vectorized<Scalar>;
        Vec kZeroVec(0.0f);
        Vec kOneSixthVec(1.0f / 6.0f);
        cpu_kernel_vec(
            iter,
            [=](Scalar grad_val, Scalar self_val) {
              return (self_val > neg_three && self_val < three)
                ? grad_val * one_sixth
                : zero;
            },
            [=](Vec grad_val, Vec self_val) {
              Vec gradNonZeroMask = (self_val > neg_three) & (self_val < three);
              return Vec::blendv(kZeroVec, grad_val * kOneSixthVec, gradNonZeroMask);
            });
      });
        */
}

pub fn hardshrink_kernel(
        iter:  &mut TensorIteratorBase,
        lambd: &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardshrink_cpu", [&] {
        auto lambd_val = lambd.to<Scalar>();
        cpu_kernel_vec(
            iter,
            [=](Scalar self_val) {
              return (self_val >= -lambd_val && self_val <= lambd_val) ? Scalar(0)
                                                                       : self_val;
            },
            [=](Vectorized<Scalar> self_val) {
              return ((self_val < -lambd_val) | (self_val > lambd_val)) & self_val;
            });
      });
        */
}

pub fn softshrink_kernel(
        iter:  &mut TensorIteratorBase,
        lambd: &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "softshrink_cpu", [&]() {
        auto lambd_val = lambd.to<Scalar>();
        cpu_kernel(iter, [=](Scalar a) -> Scalar {
          return a > lambd_val ? a - lambd_val : (a < -lambd_val ? a + lambd_val : Scalar(0));
        });
      });
        */
}

pub fn shrink_backward_kernel(
        iter:  &mut TensorIteratorBase,
        lambd: &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "shrink_backward_cpu", [&] {
        auto lambd_val = lambd.to<Scalar>();
        cpu_kernel_vec(
            iter,
            [=](Scalar grad_val, Scalar self_val) {
              return (self_val >= -lambd_val && self_val <= lambd_val) ? Scalar(0)
                                                                       : grad_val;
            },
            [=](Vectorized<Scalar> grad_val, Vectorized<Scalar> self_val) {
              return ((self_val < -lambd_val) | (self_val > lambd_val)) & grad_val;
            });
      });
        */
}

pub fn hardtanh_backward_kernel(
        iter: &mut TensorIterator,
        min:  &Scalar,
        max:  &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardshrink_backward_cpu", [&] {
        auto min_val = min.to<Scalar>();
        auto max_val = max.to<Scalar>();
        cpu_kernel_vec(
            iter,
            [=](Scalar grad_val, Scalar self_val) {
              return (self_val <= min_val || self_val >= max_val) ? Scalar(0) : grad_val;
            },
            [=](Vectorized<Scalar> grad_val, Vectorized<Scalar> self_val) {
              return ((self_val > min_val) & (self_val < max_val)) & grad_val;
            });
      });
        */
}

pub fn hardswish_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardswish_cpu", [&]() {
        const Scalar zero(0.0f);
        const Scalar three(3.0f);
        const Scalar six(6.0f);
        using Vec = vec::Vectorized<Scalar>;
        const Vec kZeroVec(zero);
        const Vec kThreeVec(three);
        const Vec kSixVec(six);
        cpu_kernel_vec(
          iter,
          [&](Scalar x) {
            return x * std::min(std::max(x + three, zero), six) / six;
          },
          [&](Vec x_vec) {
            return x_vec * vec::minimum(
              vec::maximum(x_vec + kThreeVec, kZeroVec),
              kSixVec
            ) / kSixVec;
          }
        );
      });
        */
}

pub fn hardswish_backward_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hardswish_backward_cpu", [&]() {
        const Scalar zero(0.0f);
        const Scalar three(3.0f);
        const Scalar neg_three(-3.0f);
        const Scalar one_half(0.5f);
        using Vec = vec::Vectorized<Scalar>;
        const Vec kZeroVec(zero);
        const Vec kThreeVec(three);
        const Vec kNegThreeVec(neg_three);
        const Vec kOneHalfVec(one_half);
        cpu_kernel_vec(
          iter,
          [&](Scalar grad_val, Scalar self_val) {
            if (self_val < neg_three) {
              return zero;
            } else if (self_val <= three) {
              return grad_val * ((self_val / three) + one_half);
            } else {
              return grad_val;
            }
          },
          [&](Vec grad_val, Vec self_val) {
            return Vec::blendv(
              Vec::blendv(
                grad_val * ((self_val / kThreeVec) + kOneHalfVec),
                grad_val,
                self_val >= kThreeVec
              ),
              kZeroVec,
              self_val < kNegThreeVec
            );
          }
        );
      });
        */
}

pub fn leaky_relu_kernel(
        iter:   &mut TensorIteratorBase,
        negval: &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "leaky_relu_cpu", [&] {
        using Vec = Vectorized<Scalar>;
        auto zero_vec = Vec((Scalar)(0));
        auto one_vec = Vec((Scalar)(1));
        Scalar negval = negval_.to<Scalar>();
        Vec negval_v = Vec(negval);
        cpu_kernel_vec(
            iter,
            [&](Scalar a) -> Scalar {
              return a > Scalar(0) ? a : a * negval;
            },
            [&](Vec a) -> Vec {
              auto r = Vec::blendv(negval_v, one_vec, a > zero_vec);
              return a * r;
            });
      });
        */
}

pub fn leaky_relu_backward_kernel(
        iter:   &mut TensorIteratorBase,
        negval: &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "leaky_relu_backward_cpu", [&] {
        using Vec = Vectorized<Scalar>;
        auto zero_vec = Vec((Scalar)(0));
        auto one_vec = Vec((Scalar)(1));
        Scalar negval = negval_.to<Scalar>();
        Vec negval_v = Vec(negval);
        cpu_kernel_vec(
            iter,
            [&](Scalar a, Scalar b) -> Scalar {
              return a > Scalar(0) ? b : b * negval;
            },
            [&](Vec a, Vec b) -> Vec {
              auto r = Vec::blendv(negval_v, one_vec, a > zero_vec);
              return b * r;
            });
      });
        */
}

pub fn softplus_kernel(
        iter:      &mut TensorIteratorBase,
        beta:      &Scalar,
        threshold: &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "softplus_cpu", [&]() {
        using Vec = Vectorized<Scalar>;
        auto beta = beta_.to<Scalar>();
        auto threshold = threshold_.to<Scalar>();
        const Vec beta_vec(beta);
        const Vec threshold_vec(threshold);
        cpu_kernel_vec(
            iter,
            [beta, threshold](Scalar a) -> Scalar {
              return (a * beta) > threshold ? a
                : static_cast<Scalar>(std::log1p(std::exp(a * beta))) / beta;
            },
            [beta_vec, threshold_vec](Vec a) -> Vec {
              return Vec::blendv((a * beta_vec).exp().log1p() / beta_vec, a, (a * beta_vec) > threshold_vec);
            }
        );
      });
        */
}

pub fn softplus_backward_kernel(
        iter:      &mut TensorIteratorBase,
        beta:      &Scalar,
        threshold: &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "softplus_backward_cpu", [&]() {
        using Vec = Vectorized<Scalar>;
        auto beta = beta_.to<Scalar>();
        auto threshold = threshold_.to<Scalar>();
        const Vec beta_vec(beta);
        const Vec threshold_vec(threshold);
        const Vec one_vec(static_cast<Scalar>(1.0));
        cpu_kernel_vec(
            iter,
            [beta, threshold](Scalar a, Scalar b) -> Scalar {
              Scalar z = std::exp(b * beta);
              return (b * beta) > threshold ? a : a * z / (z + Scalar(1.));
            },
            [beta_vec, one_vec, threshold_vec](Vec a, Vec b) -> Vec {
              const Vec z = (b * beta_vec).exp();
              return Vec::blendv(a * z / (z + one_vec), a, (b * beta_vec) > threshold_vec);
            }
        );
      });
        */
}

pub fn glu_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "glu_cpu", [&] {
        using Vec = Vectorized<Scalar>;
        const Scalar one_val(1);
        const Vec one_vec(one_val);
        cpu_kernel_vec(
          iter,
          [one_val](Scalar a, Scalar b) -> Scalar {
            return a * (one_val / (one_val + std::exp(-b)));
          },
          [one_vec](Vec a, Vec b) -> Vec {
            return a * (one_vec / (one_vec + b.neg().exp()));
          }
        );
      });
        */
}

pub fn glu_backward_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "glu_backward_cpu", [&] {
        using Vec = Vectorized<Scalar>;
        const Scalar one_val(1);
        const Vec one_vec(one_val);
        cpu_kernel_vec(
          iter,
          [one_val](Scalar a, Scalar b, Scalar c) -> Scalar {
            return (one_val - a) * a * b * c;
          },
          [one_vec](Vec a, Vec b, Vec c) -> Vec {
            return (one_vec - a) * a * b * c;
          }
        );
      });
        */
}

pub fn silu_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
          kBFloat16, iter.dtype(), "silu_cpu", [&]() {
            const Vectorized<Scalar> kOneVec(Scalar(1));
            cpu_kernel_vec(
                iter,
                [](Scalar x) {
                  return x / (Scalar(1) + std::exp(-x));
                },
                [kOneVec](Vectorized<Scalar> x_vec) {
                  return x_vec / (kOneVec + x_vec.neg().exp());
                });
          });
        */
}

pub fn silu_backward_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
          kBFloat16, iter.dtype(), "silu_backward_cpu", [&]() {
            const Vectorized<Scalar> kOneVec(Scalar(1));
            cpu_kernel_vec(
                iter,
                [](Scalar dy, Scalar x) {
                  const Scalar sigmoid =
                      Scalar(1) / (Scalar(1) + std::exp(-x));
                  return dy * sigmoid * (Scalar(1) + x * (Scalar(1) - sigmoid));
                },
                [kOneVec](Vectorized<Scalar> dy_vec, Vectorized<Scalar> x_vec) {
                  const Vectorized<Scalar> sigmoid =
                      kOneVec / (kOneVec + x_vec.neg().exp());
                  return dy_vec * sigmoid * (kOneVec + x_vec * (kOneVec - sigmoid));
                });
          });
        */
}

pub fn mish_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "mish_cpu", [&]() {
            using Vec = Vectorized<Scalar>;
            cpu_kernel_vec(
                iter,
                [](Scalar x) -> Scalar{
                  return static_cast<Scalar>(x * std::tanh(std::log1p(std::exp(x))));
                },
                [](Vec x_vec) -> Vec {
                  return x_vec * x_vec.exp().log1p().tanh();
                });
          });
        */
}

pub fn mish_backward_kernel(iter: &mut TensorIterator)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "mish_backward_cpu", [&]() {
            using Vec = Vectorized<Scalar>;
            const Vec kOneVec(Scalar(1));
            cpu_kernel_vec(
                iter,
                [](Scalar dy, Scalar x) -> Scalar {
                  const Scalar sigmoid =
                      Scalar(1) / (Scalar(1) + std::exp(-x));
                  const Scalar tanh_softplus = std::tanh(std::log1p(std::exp(x)));
                  return dy * (tanh_softplus + x * sigmoid * (Scalar(1) - tanh_softplus * tanh_softplus));
                },
                [kOneVec](Vec dy_vec, Vec x_vec) -> Vec {
                  const Vec sigmoid = kOneVec / (kOneVec + x_vec.neg().exp());
                  const Vec tanh_softplus = x_vec.exp().log1p().tanh();
                  return dy_vec * (tanh_softplus + x_vec * sigmoid * (kOneVec - tanh_softplus * tanh_softplus));
                });
          });
        */
}

register_dispatch!{log_sigmoid_cpu_stub          , &log_sigmoid_cpu_kernel }
register_dispatch!{log_sigmoid_backward_cpu_stub , &log_sigmoid_backward_cpu_kernel }
register_dispatch!{threshold_stub                , &threshold_kernel }
register_dispatch!{elu_stub                      , &elu_kernel }
register_dispatch!{elu_backward_stub             , &elu_backward_kernel }
register_dispatch!{GeluKernel                    , &GeluKernelImpl }
register_dispatch!{GeluBackwardKernel            , &GeluBackwardKernelImpl }
register_dispatch!{hardtanh_backward_stub        , &hardtanh_backward_kernel }
register_dispatch!{hardsigmoid_stub              , &hardsigmoid_kernel }
register_dispatch!{hardsigmoid_backward_stub     , &hardsigmoid_backward_kernel }
register_dispatch!{hardswish_stub                , &hardswish_kernel }
register_dispatch!{hardswish_backward_stub       , &hardswish_backward_kernel }
register_dispatch!{hardshrink_stub               , &hardshrink_kernel }
register_dispatch!{softshrink_stub               , &softshrink_kernel }
register_dispatch!{shrink_backward_stub          , &shrink_backward_kernel }
register_dispatch!{leaky_relu_stub               , &leaky_relu_kernel }
register_dispatch!{leaky_relu_backward_stub      , &leaky_relu_backward_kernel }
register_dispatch!{softplus_stub                 , &softplus_kernel }
register_dispatch!{softplus_backward_stub        , &softplus_backward_kernel }
register_dispatch!{glu_stub                      , &glu_kernel }
register_dispatch!{glu_backward_stub             , &glu_backward_kernel }
register_dispatch!{silu_stub                     , &silu_kernel }
register_dispatch!{silu_backward_stub            , &silu_backward_kernel }
register_dispatch!{mish_stub                     , &mish_kernel }
register_dispatch!{mish_backward_stub            , &mish_backward_kernel }
