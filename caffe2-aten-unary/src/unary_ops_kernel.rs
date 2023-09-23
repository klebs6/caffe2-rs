crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp]

pub fn sigmoid_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, iter.common_dtype(), "sigmoid_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return (static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + exp((-a)))); },
            [=](Vectorized<scalar_t> a) {
              a = Vectorized<scalar_t>(static_cast<scalar_t>(0)) - a;
              a = a.exp();
              a = Vectorized<scalar_t>(static_cast<scalar_t>(1)) + a;
              a = a.reciprocal();
              return a;
            });
      });
        */
}

#[cfg(AT_MKL_ENABLED)]
pub fn vml_log<T>(
        N: i64,
        X: *const T,
        Y: *mut T)  {

    todo!();
        /*
            constexpr i64 K = Vectorized<T>::size();
      parallel_for(0, N, K, [=](i64 begin, i64 end) {
        vec::map(
            [](Vectorized<T> x_vec) { return x_vec.log(); },
            Y + begin,
            X + begin,
            end - begin);
      });
        */
}

#[cfg(AT_MKL_ENABLED)]
pub fn vml_log_float(
        N: i64,
        X: *const f32,
        Y: *mut f32)  {
    
    todo!();
        /*
            vsLn(N, X, Y);
        */
}

#[cfg(AT_MKL_ENABLED)]
pub fn vml_log_double(
        N: i64,
        X: *const f64,
        Y: *mut f64)  {
    
    todo!();
        /*
            vdLn(N, X, Y);
        */
}

#[cfg(AT_MKL_ENABLED)]
pub fn logit_mkl_kernel<T>(
        eps: T,
        it:  *mut TensorIteratorBase)  {

    todo!();
        /*
            if (!it->can_use_32bit_indexing()) {
        for (auto& sub_it : it->with_32bit_indexing()) {
          LogitMKLKernel<T>(eps, &sub_it);
        }
        return;
      }

      constexpr i64 K = Vectorized<T>::size();
      const i64 N = it->numel();
      const T* X_data = static_cast<T*>(it->data_ptr(1));
      T* Y_data = static_cast<T*>(it->data_ptr(0));
      if (eps < T(0)) {
        parallel_for(0, N, K, [=](i64 begin, i64 end) {
          for (i64 i = begin; i < end; ++i) {
            Y_data[i] = X_data[i] == T(1) ? numeric_limits<T>::infinity()
                                          : X_data[i] / (T(1) - X_data[i]);
          }
          VmlLog<T>(end - begin, Y_data + begin, Y_data + begin);
        });
      } else {
        const T lo = eps;
        const T hi = T(1) - eps;
        parallel_for(0, N, K, [=](i64 begin, i64 end) {
          for (i64 i = begin; i < end; ++i) {
            const T x = X_data[i] < lo ? lo : (X_data[i] > hi ? hi : X_data[i]);
            Y_data[i] =
                x == T(1) ? numeric_limits<T>::infinity() : (x / (T(1) - x));
          }
          VmlLog<T>(end - begin, Y_data + begin, Y_data + begin);
        });
      }
        */
}

#[cfg(not(AT_MKL_ENABLED))]
pub fn logit_mkl_kernel<T>(
        eps: T,
        it:  *mut TensorIteratorBase)  {

    todo!();
        /*
            TORCH_CHECK(false, "ATen not compiled with MKL");
        */
}

pub fn logit_kernel(
        iter:       &mut TensorIteratorBase,
        eps_scalar: &Scalar)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND(
          kBFloat16, iter.common_dtype(), "logit_cpu", [&]() {
            const scalar_t eps = eps_scalar.to<scalar_t>();
            if (hasMKL() && iter.is_contiguous()) {
              LogitMKLKernel<scalar_t>(eps, &iter);
            } else if (eps < scalar_t(0)) {
              const Vectorized<scalar_t> kOneVec(scalar_t(1));
              cpu_kernel_vec(
                  iter,
                  [](scalar_t x) {
                    return x == scalar_t(1)
                        ? numeric_limits<scalar_t>::infinity()
                        : log(x / (scalar_t(1) - x));
                  },
                  [kOneVec](Vectorized<scalar_t> x_vec) {
                    return (x_vec / (kOneVec - x_vec)).log();
                  });
            } else {
              const scalar_t lo = eps;
              const scalar_t hi = scalar_t(1) - eps;
              const Vectorized<scalar_t> kOneVec(scalar_t(1));
              const Vectorized<scalar_t> lo_vec(lo);
              const Vectorized<scalar_t> hi_vec(hi);
              cpu_kernel_vec(
                  iter,
                  [lo, hi](scalar_t x) {
                    x = x < lo ? lo : (x > hi ? hi : x);
                    return x == scalar_t(1)
                        ? numeric_limits<scalar_t>::infinity()
                        : log(x / (scalar_t(1) - x));
                  },
                  [kOneVec, lo_vec, hi_vec](Vectorized<scalar_t> x_vec) {
                    x_vec = vec::clamp(x_vec, lo_vec, hi_vec);
                    return (x_vec / (kOneVec - x_vec)).log();
                  });
            }
          });
        */
}

pub fn abs_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "abs_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return abs_impl(a); },
            [=](Vectorized<scalar_t> a) { return a.abs(); });
      });
        */
}

pub fn angle_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "angle_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return angle_impl(a); },
            [=](Vectorized<scalar_t> a) { return a.angle(); });
      });
        */
}

pub fn real_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "real_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return real_impl(a); },
            [=](Vectorized<scalar_t> a) { return a.real(); });
      });
        */
}

pub fn imag_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "imag_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return imag_impl(a); },
            [=](Vectorized<scalar_t> a) { return a.imag(); });
      });
        */
}

// NB: Ignores the negative bit on tensors
pub fn conj_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          kBool, kBFloat16, kHalf, iter.common_dtype(), "conj_cpu", [&]() {
            cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t { return conj_impl(a); },
                [=](Vectorized<scalar_t> a) { return a.conj(); });
          });
        */
}

pub fn bitwise_not_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            if (iter.dtype() == ScalarType::Bool) {
        // Boolean type does not work with ~ (bitwise NOT) in C++. bitwise_not wraps this operation for both Boolean and
        // integral types.
        cpu_kernel(
              iter,
              [](bool a) {
                return !a;
              });
      } else {
        AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_cpu", [&]() {
          cpu_kernel_vec(
              iter,
              [](scalar_t a) -> scalar_t {
                return ~a;
              },
              [](Vectorized<scalar_t> a) -> Vectorized<scalar_t> {
                return ~a;
              });
        });
      }
        */
}

pub fn frac_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "frac_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return a - trunc(a); },
            [=](Vectorized<scalar_t> a) { return a.frac(); });
      });
        */
}

pub fn logical_not_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            // NOTE: this implementation differs from the CUDA implementation which only does single dispatch
      // (to avoid expensive compilation) because CPU kernels don't handle dynamic_casting
      // (see needs_dynamic_casting).
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(1), "logical_not_cpu", [&]() {
        using self_t = scalar_t;
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(0), "logical_not_cpu", [&]() {
          cpu_kernel(iter, [](self_t a) -> scalar_t { return static_cast<scalar_t>(!a); });
        });
      });
        */
}

pub fn reciprocal_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "reciprocal_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) __ubsan_ignore_float_divide_by_zero__ -> scalar_t { return static_cast<scalar_t>(1.0) / a; },
            [=](Vectorized<scalar_t> a) { return a.reciprocal(); });
      });
        */
}

pub fn neg_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "neg_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return -a; },
            [=](Vectorized<scalar_t> a) { return a.neg(); });
      });
        */
}

pub fn sign_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            if(iter.dtype() == ScalarType::Bool){
          cpu_kernel(iter, [=](bool x) -> bool { return x; });
      } else {
        AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, ScalarType::Half, iter.dtype(), "sign_cpu", [&]() {
            auto zero_vec = Vectorized<scalar_t>(static_cast<scalar_t>(0));
            auto one_vec = Vectorized<scalar_t>(static_cast<scalar_t>(1));

            cpu_kernel_vec(
              iter,
              [=](scalar_t a) -> scalar_t { return (0 < a) - (a < 0); },
              [=](Vectorized<scalar_t> self_vec){

                  // Comparison operators returns bitmask.
                  auto left = Vectorized<scalar_t>::blendv(zero_vec, one_vec, zero_vec < self_vec);
                  auto right = Vectorized<scalar_t>::blendv(zero_vec, one_vec, self_vec < zero_vec);

                  return left - right;
              });
        });
      }
        */
}

pub fn signbit_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, ScalarType::Half, iter.input_dtype(), "signbit_cpu", [&]() {
        cpu_kernel(iter, [](scalar_t a) -> bool { return a < 0; });
      });
        */
}

pub fn sgn_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "sgn_cpu", [&]() {
        cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return sgn_impl(a); },
          [=](Vectorized<scalar_t> a) { return a.sgn(); });
      });
        */
}

pub fn sinc_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, iter.common_dtype(), "sinc_cpu", [&]() {
        cpu_kernel(
            iter,
            [=](scalar_t a) -> scalar_t {
              if (a == scalar_t(0)) {
                return scalar_t(1);
              } else {
                scalar_t product = pi<scalar_t> * a;
                return sin(product) / product;
              }
            });
      });
        */
}

pub fn sinh_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "sinh_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return sinh(a); },
            [=](Vectorized<scalar_t> self_vec){return self_vec.sinh();});
      });
        */
}

pub fn cosh_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "cosh_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return cosh(a); },
            [=](Vectorized<scalar_t> self_vec){return self_vec.cosh();});
      });
        */
}

pub fn acosh_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "acosh_cpu", [&]() {
          cpu_kernel(
            iter,
            [=](scalar_t a) -> scalar_t { return acosh(a); });
        });
        */
}

pub fn asinh_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "asinh_cpu", [&]() {
          cpu_kernel(
            iter,
            [=](scalar_t a) -> scalar_t { return asinh(a); });
        });
        */
}

pub fn atanh_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "atanh_cpu", [&]() {
          cpu_kernel(
            iter,
            [=](scalar_t a) -> scalar_t { return atanh(a); });
        });
        */
}

pub fn digamma_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "digamma", [&]() {
        cpu_kernel(
            iter,
            [=](scalar_t a) -> scalar_t { return calc_digamma(a); });
      });
        */
}

pub fn trigamma_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "trigamma", [&]() {
        cpu_kernel(
            iter,
            [=](scalar_t a) -> scalar_t { return trigamma(a); });
      });
        */
}

pub fn exp2_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            // Supports only floating types as exp2 doesn't have
      // complex overloads.
      AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.dtype(), "exp2", [&]() {
        cpu_kernel(
            iter,
            [=](scalar_t a) -> scalar_t { return exp2(a); });
      });
        */
}

pub fn polygamma_kernel(
        iter: &mut TensorIteratorBase,
        n:    i64)  {
    
    todo!();
        /*
            if (n == 0) {
        digamma_kernel(iter);
      } else if (n == 1) {
        trigamma_kernel(iter);
      } else {
        AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "polygamma", [&]() {
          cpu_kernel(
              iter, [=](scalar_t a) -> scalar_t { return calc_polygamma(n, a); });
        });
      }
        */
}

pub fn nan_to_num_kernel(
        iter:    &mut TensorIteratorBase,
        nan:     Option<f64>,
        pos_inf: Option<f64>,
        neg_inf: Option<f64>)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.dtype(), "nan_to_num", [&]() {
        scalar_t nan_replacement = static_cast<scalar_t>(nan.value_or(0.));
        scalar_t pos_inf_replacement = pos_inf.has_value()
            ? static_cast<scalar_t>(pos_inf.value())
            : scalar_t::max;
        scalar_t neg_inf_replacement = neg_inf.has_value()
            ? static_cast<scalar_t>(neg_inf.value())
            : numeric_limits<scalar_t>::lowest();

        cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
          return (
              _isnan(a)
                  ? nan_replacement
                  : (a == numeric_limits<scalar_t>::infinity()
                         ? pos_inf_replacement
                         : (a == -numeric_limits<scalar_t>::infinity()
                                ? neg_inf_replacement
                                : a)));
        });
      });
        */
}

pub fn kaiser_window_kernel(
        iter:          &mut TensorIteratorBase,
        window_length: i64,
        beta:          f64)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), "kaiser_window_cpu", [&](){
        const scalar_t alpha = static_cast<scalar_t>((window_length - 1) / 2.0);
        cpu_kernel(iter, [=](scalar_t a){
            return calc_i0(static_cast<scalar_t>(beta) * sqrt(1 - pow((a - alpha) / alpha, static_cast<scalar_t>(2.0)))) / calc_i0(static_cast<scalar_t>(beta));
        });
      });
        */
}

pub fn cauchy_kernel(
        iter:   &mut TensorIteratorBase,
        median: f64,
        sigma:  f64,
        gen:    Option<Generator>)  {
    
    todo!();
        /*
            CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
      templates::cpu::cauchy_kernel(iter, median, sigma, generator);
        */
}

pub fn bernoulli_tensor_kernel(
        self_: &mut Tensor,
        p:     &Tensor,
        gen:   Option<Generator>)  {
    
    todo!();
        /*
            CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
      templates::cpu::bernoulli_kernel(self, p_, generator);
        */
}

pub fn bernoulli_scalar_kernel_default(
        self_: &mut Tensor,
        p:     f64,
        gen:   Option<Generator>)  {
    
    todo!();
        /*
            CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
      templates::cpu::bernoulli_kernel(self, p, generator);
        */
}

#[cfg(not(AT_MKL_ENABLED))]
pub fn bernoulli_scalar_kernel(
        self_: &mut Tensor,
        p:     f64,
        gen:   Option<Generator>)  {
    
    todo!();
        /*
            bernoulli_scalar_kernel_default(self, p, gen);
        */
}

#[cfg(AT_MKL_ENABLED)]
pub fn bernoulli_scalar_kernel(
        self_: &mut Tensor,
        p:     f64,
        gen:   Option<Generator>)  {
    
    todo!();
        /*
            if (cpuinfo_initialize() && cpuinfo_vendor_intel == cpuinfo_get_processor(0)->core->vendor) {
        CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
        i64 seed;
        {
          // See Note [Acquire lock when using random generators]
          lock_guard<mutex> lock(generator->mutex_);
          seed = generator->random();
        }
        i64 n = self.numel();
        bool contig = self.is_contiguous();

        AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "bernoulli_scalar_cpu_", [&] {
          Tensor tmp_int_tensor;
          if (is_same<scalar_t, int>::value && contig) {
            tmp_int_tensor = self;
          } else {
            tmp_int_tensor = empty(self.sizes(), self.options().dtype(kInt));
          }

          scalar_t *self_ptr = self.data_ptr<scalar_t>();
          int *sample_int_ptr = tmp_int_tensor.data_ptr<int>();

          auto sample = [&](i64 begin, i64 end) {
            i64 len = end - begin;
            if (len > 0) {
              VSLStreamStatePtr stream;
              vslNewStream(&stream, VSL_BRNG_MCG31, seed);
              vslSkipAheadStream(stream, begin);
              viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, len,
                sample_int_ptr + begin, p);
              vslDeleteStream(&stream);

              // vectorized copy if using buffer and contiguous, i.e., being non-int
              // type and contiguous
              if (!is_same<scalar_t, int>::value && contig) {
                scalar_t *self_seg = self_ptr + begin;
                int* tmp_seg = sample_int_ptr + begin;
                vec::convert<int, scalar_t>(tmp_seg, self_seg, len);
              }
            }
          };

          parallel_for(0, n, /* grain_size= */ 800, sample);

          // copy_ if using buffer and non contiguous
          if (!contig) {
            self.copy_(tmp_int_tensor);
          }
        });
      } else {
        // The situation of AMD, move to using the default version
        bernoulli_scalar_kernel_default(self, p, gen);
      }
        */
}

pub fn exponential_kernel(
        iter:   &mut TensorIteratorBase,
        lambda: f64,
        gen:    Option<Generator>)  {
    
    todo!();
        /*
            CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
      templates::cpu::exponential_kernel(iter, lambda, generator);
        */
}

pub fn geometric_kernel(
        iter: &mut TensorIteratorBase,
        p:    f64,
        gen:  Option<Generator>)  {
    
    todo!();
        /*
            CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
      templates::cpu::geometric_kernel(iter, p, generator);
        */
}

pub fn log_normal_kernel(
        iter: &mut TensorIteratorBase,
        mean: f64,
        std:  f64,
        gen:  Option<Generator>)  {
    
    todo!();
        /*
            CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
      templates::cpu::log_normal_kernel(iter, mean, std, generator);
        */
}

pub fn uniform_kernel(
        iter: &mut TensorIteratorBase,
        from: f64,
        to:   f64,
        gen:  Option<Generator>)  {
    
    todo!();
        /*
            CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
      templates::cpu::uniform_kernel(iter, from, to, generator);
        */
}

pub fn normal_kernel(
        self_: &mut Tensor,
        mean:  f64,
        std:   f64,
        gen:   Option<Generator>)  {
    
    todo!();
        /*
            CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
      templates::cpu::normal_kernel(self, mean, std, generator);
        */
}

pub fn random_from_to_kernel(
        iter:  &mut TensorIteratorBase,
        range: u64,
        base:  i64,
        gen:   Option<Generator>)  {
    
    todo!();
        /*
            CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
      templates::cpu::random_from_to_kernel(iter, range, base, generator);
        */
}

pub fn random_kernel(
        iter: &mut TensorIteratorBase,
        gen:  Option<Generator>)  {
    
    todo!();
        /*
            CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
      templates::cpu::random_kernel(iter, generator);
        */
}

/**
  | This is the special kernel to handle single
  | specific case:
  |
  | from(inclusive)
  | = numeric_limits<i64>::lowest()
  |
  | to(exclusive) = None (=
  | i64::max + 1)
  |
  */
pub fn random_full_64_bits_range_kernel(
        iter: &mut TensorIteratorBase,
        gen:  Option<Generator>)  {
    
    todo!();
        /*
            CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
      templates::cpu::random_full_64_bits_range_kernel(iter, generator);
        */
}

pub fn rsqrt_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, iter.common_dtype(), "rsqrt_cpu", [&] {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
              return (static_cast<scalar_t>(1)) / sqrt(a);
            },
            [=](Vectorized<scalar_t> a) { return a.rsqrt(); });
      });
        */
}

pub fn entr_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND(
          kBFloat16, iter.common_dtype(), "entr_cpu", [&] {
            cpu_kernel(iter, [](scalar_t x) -> scalar_t {
              if (_isnan(x)) {
                return x;
              } else if (x > 0) {
                return -x * log(x);
              } else if (x == 0) {
                return static_cast<scalar_t>(0);
              }
              return static_cast<scalar_t>(-INFINITY);
            });
          });
        */
}

pub fn frexp_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND(kHalf,
        // The iter.dtype() here is the dtype of mantissa output.
        // It's a floating point type and must be the same as the input's dtype.
        iter.dtype(),
        "frexp_cpu", [&]() {
          cpu_kernel_multiple_outputs(
            iter,
            [](scalar_t a) -> tuple<scalar_t, i32> {
              i32 exponent;
              scalar_t mantissa = frexp(a, &exponent);
              return tuple<scalar_t, i32>(mantissa, exponent);
            }
          );
      });
        */
}

pub fn i0e_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
      AT_DISPATCH_FLOATING_TYPES_AND(
          kBFloat16, iter.common_dtype(), "i0e_cpu", [&]() {
            cpu_kernel_vec(
                iter,
                [](scalar_t x) { return calc_i0e(x); },
                [](Vectorized<scalar_t> x) { return x.i0e(); });
          });
        */
}

pub fn i1_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
      AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "i1_cpu", [&]() {
        cpu_kernel(iter, [](scalar_t x) { return calc_i1(x); });
      });
        */
}

pub fn i1e_kernel(iter: &mut TensorIteratorBase)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
      AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "i1e_cpu", [&]() {
        cpu_kernel(iter, [](scalar_t x) { return calc_i1e(x); });
      });
        */
}

/**
  | TODO: Disable cont. branch to test more
  | risky code
  |
  */
#[macro_export] macro_rules! implement_iterator_lambda {
    ($op:ident) => {
        /*
        
                  [&](char** data_, const i64* strides, i64 n) {              
                    scalar_t* out_data = reinterpret_cast<scalar_t*>(data_[0]);       
                    scalar_t* in_data = reinterpret_cast<scalar_t*>(data_[1]);        
                    i64 out_stride = strides[0] / sizeof(scalar_t);               
                    i64 in_stride = strides[1] / sizeof(scalar_t);                
                    if (out_stride == 1 && in_stride == 1) {                          
                      vml::v##op(out_data, in_data, n);                               
                    } else {                                                          
                      static constexpr i64 WIDTH = 131072 / sizeof(scalar_t);     
                      for (i64 i = 0; i < n; i += WIDTH) {                        
                        scalar_t buffer[WIDTH];                                       
                        i64 width = WIDTH;                                        
                        width = min(width, n - i);                               
                        for (i64 j = 0; j < width; j++)                           
                          buffer[j] = in_data[in_stride * (i + j)];                   
                        vml::v##op(buffer, buffer, width);                            
                        for (i64 j = 0; j < width; j++)                           
                          out_data[out_stride * (i + j)] = buffer[j];                 
                      }                                                               
                    }                                                                 
                  }
        */
    }
}

#[macro_export] macro_rules! implement_float_kernel {
    ($op:ident) => {
        /*
        
          namespace CPU_CAPABILITY {                                                        
          void op##_kernel(TensorIteratorBase& iter) {                                      
            TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                                    
            AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), #op "_vml_cpu", [&]() { 
              iter.serial_for_each(                                                         
                  IMPLEMENT_ITERATOR_LAMBDA(op),                                            
                  {0, iter.numel()});                                                       
            });                                                                             
            iter.cast_outputs();                                                            
          }                                                                                 
          }                                                                                 
          register_dispatch(op##_stub, &CPU_CAPABILITY::op##_kernel)
        */
    }
}

#[macro_export] macro_rules! implement_complex_kernel {
    ($op:ident) => {
        /*
        
          namespace CPU_CAPABILITY {                                                                     
          void op##_kernel(TensorIteratorBase& iter) {                                                   
            TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                                                 
            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kBFloat16, iter.dtype(), #op "_vml_cpu", [&]() { 
              iter.serial_for_each(                                                                      
                  IMPLEMENT_ITERATOR_LAMBDA(op),                                                         
                  {0, iter.numel()});                                                                    
            });                                                                                          
            iter.cast_outputs();                                                                         
          }                                                                                              
          }                                                                                              
          register_dispatch(op##_stub, &CPU_CAPABILITY::op##_kernel)
        */
    }
}

register_dispatch!{rsqrt_stub                     , &CPU_CAPABILITY::rsqrt_kernel}
register_dispatch!{sigmoid_stub                   , &CPU_CAPABILITY::sigmoid_kernel}
register_dispatch!{logit_stub                     , &CPU_CAPABILITY::logit_kernel}
register_dispatch!{bernoulli_tensor_stub          , &CPU_CAPABILITY::bernoulli_tensor_kernel}
register_dispatch!{bernoulli_scalar_stub          , &CPU_CAPABILITY::bernoulli_scalar_kernel}
register_dispatch!{cauchy_stub                    , &CPU_CAPABILITY::cauchy_kernel}
register_dispatch!{exponential_stub               , &CPU_CAPABILITY::exponential_kernel}
register_dispatch!{geometric_stub                 , &CPU_CAPABILITY::geometric_kernel}
register_dispatch!{log_normal_stub                , &CPU_CAPABILITY::log_normal_kernel}
register_dispatch!{normal_stub                    , &CPU_CAPABILITY::normal_kernel}
register_dispatch!{uniform_stub                   , &CPU_CAPABILITY::uniform_kernel}
register_dispatch!{random_from_to_stub            , &CPU_CAPABILITY::random_from_to_kernel}
register_dispatch!{random_full_64_bits_range_stub , &CPU_CAPABILITY::random_full_64_bits_range_kernel}
register_dispatch!{random_stub                    , &CPU_CAPABILITY::random_kernel}
register_dispatch!{abs_stub                       , &CPU_CAPABILITY::abs_kernel}
register_dispatch!{angle_stub                     , &CPU_CAPABILITY::angle_kernel}
register_dispatch!{real_stub                      , &CPU_CAPABILITY::real_kernel}
register_dispatch!{imag_stub                      , &CPU_CAPABILITY::imag_kernel}
register_dispatch!{conj_physical_stub             , &CPU_CAPABILITY::conj_kernel}
register_dispatch!{exp2_stub                      , &CPU_CAPABILITY::exp2_kernel}
register_dispatch!{bitwise_not_stub               , &CPU_CAPABILITY::bitwise_not_kernel}
register_dispatch!{logical_not_stub               , &CPU_CAPABILITY::logical_not_kernel}
register_dispatch!{frac_stub                      , &CPU_CAPABILITY::frac_kernel}
register_dispatch!{reciprocal_stub                , &CPU_CAPABILITY::reciprocal_kernel}
register_dispatch!{nan_to_num_stub                , &CPU_CAPABILITY::nan_to_num_kernel}
register_dispatch!{neg_stub                       , &CPU_CAPABILITY::neg_kernel}
register_dispatch!{sign_stub                      , &CPU_CAPABILITY::sign_kernel}
register_dispatch!{signbit_stub                   , &CPU_CAPABILITY::signbit_kernel}
register_dispatch!{sgn_stub                       , &CPU_CAPABILITY::sgn_kernel}
register_dispatch!{sinc_stub                      , &CPU_CAPABILITY::sinc_kernel}
register_dispatch!{sinh_stub                      , &CPU_CAPABILITY::sinh_kernel}
register_dispatch!{cosh_stub                      , &CPU_CAPABILITY::cosh_kernel}
register_dispatch!{acosh_stub                     , &CPU_CAPABILITY::acosh_kernel}
register_dispatch!{asinh_stub                     , &CPU_CAPABILITY::asinh_kernel}
register_dispatch!{atanh_stub                     , &CPU_CAPABILITY::atanh_kernel}
register_dispatch!{digamma_stub                   , &CPU_CAPABILITY::digamma_kernel}
register_dispatch!{trigamma_stub                  , &CPU_CAPABILITY::trigamma_kernel}
register_dispatch!{polygamma_stub                 , &CPU_CAPABILITY::polygamma_kernel}
register_dispatch!{kaiser_window_stub             , &CPU_CAPABILITY::kaiser_window_kernel}
register_dispatch!{special_entr_stub              , &CPU_CAPABILITY::entr_kernel}
register_dispatch!{frexp_stub                     , &CPU_CAPABILITY::frexp_kernel}
register_dispatch!{special_i0e_stub               , &CPU_CAPABILITY::i0e_kernel}
register_dispatch!{special_i1_stub                , &CPU_CAPABILITY::i1_kernel}
register_dispatch!{special_i1e_stub               , &CPU_CAPABILITY::i1e_kernel}

implement_complex_kernel!{acos}
implement_complex_kernel!{asin}
implement_complex_kernel!{atan}
implement_float_kernel!{ceil}
implement_complex_kernel!{cos}
implement_float_kernel!{erf}
implement_float_kernel!{erfc}
implement_float_kernel!{erfinv}
implement_complex_kernel!{exp}
implement_float_kernel!{expm1}
implement_float_kernel!{floor}
implement_complex_kernel!{log}
implement_complex_kernel!{log10}
implement_float_kernel!{log1p}
implement_complex_kernel!{log2}
implement_float_kernel!{i0}
implement_float_kernel!{round}
implement_complex_kernel!{sin}
implement_complex_kernel!{sqrt}
implement_complex_kernel!{tan}
implement_complex_kernel!{tanh}
implement_float_kernel!{trunc}
implement_float_kernel!{lgamma}
