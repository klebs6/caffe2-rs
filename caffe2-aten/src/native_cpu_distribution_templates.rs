crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/DistributionTemplates.h]

// ==================================================== Random ========================================================

pub fn random_from_to_kernel<R: Rng>(
        iter:      &mut TensorIteratorBase,
        range:     u64,
        base:      i64,
        generator: R)  {

    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "random_from_to_kernel_cpu", [&] {
        lock_guard<mutex> lock(generator->mutex_);
        cpu_serial_kernel(iter, [range, base, generator]() -> scalar_t {
          uniform_int_from_to_distribution<scalar_t> random(range, base);
          return random(generator);
        });
      });
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
pub fn random_full_64_bits_range_kernel<R: Rng>(
        iter:      &mut TensorIteratorBase,
        generator: R)  {

    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(ScalarType::BFloat16, iter.dtype(), "random_full_64_bits_range_kernel_cpu", [&] {
        lock_guard<mutex> lock(generator->mutex_);
        if (is_same<scalar_t, i64>::value ||
            is_same<scalar_t, double>::value ||
            is_same<scalar_t, float>::value ||
            is_same<scalar_t, BFloat16>::value) {
          cpu_serial_kernel(iter, [generator]() -> scalar_t {
            uniform_int_full_range_distribution<scalar_t> random;
            return random(generator);
          });
        } else {
          TORCH_CHECK(false, "random_full_64_bits_range_kernel_cpu handles only int64, double, float and bfloat16");
        }
      });
        */
}

pub struct RandomFromToKernel<R: Rng> {

}

impl<R: Rng> RandomFromToKernel<R> {
    
    pub fn invoke(&mut self, 
        iter:  &mut TensorIteratorBase,
        range: u64,
        base:  i64,
        gen:   Option<Generator>)  {
        
        todo!();
        /*
            random_from_to_kernel(iter, range, base, check_generator<R>(gen));
        */
    }
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        gen:  Option<Generator>)  {
        
        todo!();
        /*
            random_full_64_bits_range_kernel(iter, check_generator<R>(gen));
        */
    }
}

pub fn random_kernel<R: Rng>(
    iter:      &mut TensorIteratorBase,
    generator: R)  {

    todo!();
        /*
            lock_guard<mutex> lock(generator->mutex_);
      AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, iter.dtype(), "random_kernel_cpu", [&] {
        cpu_serial_kernel(iter, [generator]() -> scalar_t {
          uniform_int_distribution<scalar_t> random;
          return random(generator);
        });
      });
        */
}

pub struct RandomKernel<R: Rng> {

}

impl<R: Rng> RandomKernel<R> {
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        gen:  Option<Generator>)  {
        
        todo!();
        /*
            random_kernel(iter, check_generator<R>(gen));
        */
    }
}

// ==================================================== Normal ========================================================

#[cfg(target_feature = "avx2")]
pub fn normal_fill_16_avx2(
        data:      *mut f32,
        two_pi:    *const __m256,
        one:       *const __m256,
        minus_two: *const __m256,
        mean:      *const __m256,
        std_v:     *const __m256)  {
    
    todo!();
        /*
            const __m256 u1 = _mm256_sub_ps(*one, _mm256_loadu_ps(data));
      const __m256 u2 = _mm256_loadu_ps(data + 8);
      // sincos256_ps and log256_ps are from avx_mathfun.h
      const __m256 radius = _mm256_sqrt_ps(_mm256_mul_ps(*minus_two, log256_ps(u1)));
      const __m256 theta = _mm256_mul_ps(*two_pi, u2);
      __m256 sintheta, costheta;
      sincos256_ps(theta, &sintheta, &costheta);
      const __m256 n1 = _mm256_mul_ps(radius, costheta);
      const __m256 n2 = _mm256_mul_ps(radius, sintheta);
      _mm256_storeu_ps(data, _mm256_fmadd_ps(n1, *std_v, *mean));
      _mm256_storeu_ps(data + 8, _mm256_fmadd_ps(n2, *std_v, *mean));
        */
}

#[cfg(target_feature = "avx2")]
pub fn normal_fill_avx2<R: Rng>(
    self_:     &mut Tensor,
    mean:      f32,
    std:       f32,
    generator: R)  {

    todo!();
        /*
            float *data = self.data_ptr<float>();
      auto size = self.numel();
      lock_guard<mutex> lock(generator->mutex_);
      for (i64 i = 0; i < size; ++i) {
        uniform_real_distribution<float> uniform(0, 1);
        data[i] = uniform(generator);
      }
      const __m256 two_pi = _mm256_set1_ps(2.0f * pi<double>);
      const __m256 one = _mm256_set1_ps(1.0f);
      const __m256 minus_two = _mm256_set1_ps(-2.0f);
      const __m256 mean_v = _mm256_set1_ps(mean);
      const __m256 std_v = _mm256_set1_ps(std);

      for (i64 i = 0; i < size - 15; i += 16) {
        normal_fill_16_AVX2(data + i, &two_pi, &one, &minus_two, &mean_v, &std_v);
      }

      if (size % 16 != 0) {
        // Recompute the last 16 values.
        data = data + size - 16;
        for (i64 i = 0; i < 16; ++i) {
          uniform_real_distribution<float> uniform(0, 1);
          data[i] = uniform(generator);
        }
        normal_fill_16_AVX2(data, &two_pi, &one, &minus_two, &mean_v, &std_v);
      }
        */
}

pub fn normal_fill_16<Scalar>(
    data: *mut Scalar,
    mean: Scalar,
    std:  Scalar)  {

    todo!();
        /*
            for (int j = 0; j < 8; ++j) {
        const scalar_t u1 = 1 - data[j]; // [0, 1) -> (0, 1] for log.
        const scalar_t u2 = data[j + 8];
        const scalar_t radius = sqrt(-2 * log(u1));
        const scalar_t theta = 2.0f * pi<double> * u2;
        data[j] = radius * cos(theta) * std + mean;
        data[j + 8] = radius * sin(theta) * std + mean;
      }
        */
}

pub fn normal_fill<Scalar, R: Rng>(
    self_:     &mut Tensor,
    mean:      Scalar,
    std:       Scalar,
    generator: R)  {

    todo!();
        /*
            scalar_t *data = self.data_ptr<scalar_t>();
      auto size = self.numel();
      lock_guard<mutex> lock(generator->mutex_);
      for (i64 i = 0; i < size; ++i) {
        uniform_real_distribution<scalar_t> uniform(0, 1);
        data[i] = uniform(generator);
      }

      for (i64 i = 0; i < size - 15; i += 16) {
        normal_fill_16<scalar_t>(data + i, mean, std);
      }
      if (size % 16 != 0) {
        // Recompute the last 16 values.
        data = data + size - 16;
        for (i64 i = 0; i < 16; ++i) {
          uniform_real_distribution<scalar_t> uniform(0, 1);
          data[i] = uniform(generator);
        }
        normal_fill_16<scalar_t>(data, mean, std);
      }
        */
}

pub fn normal_kernel<R: Rng>(
    self_:     &mut Tensor,
    mean:      f64,
    std:       f64,
    generator: R)  {

    todo!();
        /*
            auto size = self.numel();
      if (self.scalar_type() == ScalarType::Float && size >= 16 && self.is_contiguous()) {
    #ifdef target_feature = "avx2"
        normal_fill_AVX2(self, static_cast<float>(mean), static_cast<float>(std), generator);
    #else
        normal_fill(self, static_cast<float>(mean), static_cast<float>(std), generator);
    #endif
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "normal_kernel_cpu", [&] {
          if (size >= 16 && self.is_contiguous()) {
            normal_fill<scalar_t>(self, static_cast<scalar_t>(mean), static_cast<scalar_t>(std), generator);
          } else {
            auto iter = TensorIterator::borrowing_nullary_op(self);
            lock_guard<mutex> lock(generator->mutex_);
            cpu_serial_kernel(iter, [mean, std, generator]() -> scalar_t {
              normal_distribution<double> normal(mean, std);
              return static_cast<scalar_t>(normal(generator));
            });
          }
        });
      }
        */
}

pub struct NormalKernel<R: Rng> {

}

impl<R: Rng> NormalKernel<R> {
    
    pub fn invoke(&mut self, 
        self_: &mut Tensor,
        mean:  f64,
        std:   f64,
        gen:   Option<Generator>)  {
        
        todo!();
        /*
            normal_kernel(self, mean, std, check_generator<R>(gen));
        */
    }
}

// ==================================================== Uniform =======================================================

pub fn uniform_kernel<R: Rng>(
    iter:      &mut TensorIteratorBase,
    from:      f64,
    to:        f64,
    generator: R)  {

    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "uniform_kernel_cpu", [&]() {
        lock_guard<mutex> lock(generator->mutex_);
        auto from = static_cast<scalar_t>(from_);
        auto to = static_cast<scalar_t>(to_);
        uniform_real_distribution<scalar_t> uniform(from, to);
        cpu_serial_kernel(iter, [&uniform, generator]() -> scalar_t {
          return static_cast<scalar_t>(uniform(generator));
        });
      });
        */
}

pub struct UniformKernel<R: Rng> {

}

impl<R: Rng> UniformKernel<R> {
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        from: f64,
        to:   f64,
        gen:  Option<Generator>)  {
        
        todo!();
        /*
            uniform_kernel(iter, from, to, check_generator<R>(gen));
        */
    }
}

// ==================================================== Cauchy ========================================================

pub fn cauchy_kernel<R: Rng>(
    iter:      &mut TensorIteratorBase,
    median:    f64,
    sigma:     f64,
    generator: R)  {

    todo!();
    /*
            AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "cauchy_cpu", [&]() {
        lock_guard<mutex> lock(generator->mutex_);
        cauchy_distribution<double> cauchy(median, sigma);
        cpu_serial_kernel(iter, [&cauchy, generator]() -> scalar_t {
          return static_cast<scalar_t>(cauchy(generator));
        });
      });
        */
}

pub struct CauchyKernel<R: Rng> {

}

impl<R: Rng> CauchyKernel<R> {
    
    pub fn invoke(&mut self, 
        iter:   &mut TensorIteratorBase,
        median: f64,
        sigma:  f64,
        gen:    Option<Generator>)  {
        
        todo!();
        /*
            cauchy_kernel(iter, median, sigma, check_generator<R>(gen));
        */
    }
}

// ================================================== LogNormal =======================================================

pub fn log_normal_kernel<R: Rng>(
    iter:      &mut TensorIteratorBase,
    mean:      f64,
    std:       f64,
    generator: R)  {

    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "log_normal_cpu", [&]() {
        lock_guard<mutex> lock(generator->mutex_);
        lognormal_distribution<double> logNormal(mean, std);
        cpu_serial_kernel(iter, [&logNormal, generator]() -> scalar_t {
          return static_cast<scalar_t>(logNormal(generator));
        });
      });
        */
}

pub struct LogNormalKernel<R: Rng> {

}

impl<R: Rng> LogNormalKernel<R> {
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        mean: f64,
        std:  f64,
        gen:  Option<Generator>)  {
        
        todo!();
        /*
            log_normal_kernel(iter, mean, std, check_generator<R>(gen));
        */
    }
}

// =================================================== Geometric ======================================================

pub fn geometric_kernel<R: Rng>(
    iter:      &mut TensorIteratorBase,
    p:         f64,
    generator: R)  {

    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "geometric_cpu", [&]() {
        lock_guard<mutex> lock(generator->mutex_);
        geometric_distribution<double> geometric(p);
        cpu_serial_kernel(iter, [&geometric, generator]() -> scalar_t {
          return static_cast<scalar_t>(geometric(generator));
        });
      });
        */
}

pub struct GeometricKernel<R: Rng> {

}

impl<R> GeometricKernel<R: Rng> {

    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        p:    f64,
        gen:  Option<Generator>)  {
        
        todo!();
        /*
            geometric_kernel(iter, p, check_generator<R>(gen));
        */
    }
}

// ================================================== Exponential =====================================================

pub fn exponential_kernel<R: Rng>(
    iter:      &mut TensorIteratorBase,
    lambda:    f64,
    generator: R) {

    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "exponential_cpu", [&]() {
        lock_guard<mutex> lock(generator->mutex_);
        exponential_distribution<double> exponential(lambda);
        cpu_serial_kernel(iter, [&exponential, generator]() -> scalar_t {
          return static_cast<scalar_t>(exponential(generator));
        });
      });
        */
}

pub struct ExponentialKernel<R: Rng> {

}

impl<R: Rng> ExponentialKernel<R> {

    pub fn invoke(&mut self, 
        iter:   &mut TensorIteratorBase,
        lambda: f64,
        gen:    Option<Generator>)  {
        
        todo!();
        /*
            exponential_kernel(iter, lambda, check_generator<R>(gen));
        */
    }
}

// ================================================== Bernoulli =======================================================

pub fn bernoulli_kernel_tensor<R: Rng>(
    self_:     &mut Tensor,
    p:         &Tensor,
    generator: R)  {

    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "bernoulli_tensor_cpu_self_", [&] {
        // See Note [Acquire lock when using random generators]
        lock_guard<mutex> lock(generator->mutex_);
        using self_t = scalar_t;
        auto p_cpu = p_.to(kCPU);
        MaybeOwned<Tensor> p = expand_inplace(self, p_cpu);
        auto iter = TensorIteratorConfig()
            .add_output(self)
            .add_input(*p)
            .check_all_same_dtype(false)
            .build();
        if (p_.scalar_type() == kDouble) {
          cpu_serial_kernel(iter, [&](const double p_val) -> self_t {
            bernoulli_distribution<double> bernoulli(p_val);
            return static_cast<self_t>(bernoulli(generator));
          });
        } else {
          AT_DISPATCH_FLOATING_TYPES(p_.scalar_type(), "bernoulli_tensor_cpu_p_", [&] {
            using p_t = scalar_t;
            cpu_serial_kernel(iter, [&](const p_t p_val) -> self_t {
              bernoulli_distribution<float> bernoulli(p_val);
              return static_cast<self_t>(bernoulli(generator));
            });
          });
        }
      });
        */
}

pub fn bernoulli_kernel<R: Rng>(
    self_:     &mut Tensor,
    p:         f64,
    generator: R)  {

    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "bernoulli_scalar_cpu_", [&] {
        // See Note [Acquire lock when using random generators]
        lock_guard<mutex> lock(generator->mutex_);
        auto iter = TensorIterator::borrowing_nullary_op(self);
        cpu_serial_kernel(iter, [p, generator]() -> scalar_t {
          bernoulli_distribution<double> bernoulli(p);
          return static_cast<scalar_t>(bernoulli(generator));
        });
      });
        */
}

pub struct BernoulliKernel<R: Rng> {

}

impl<R: Rng> BernoulliKernel<R> {
    
    pub fn invoke(&mut self, 
        self_: &mut Tensor,
        p:     f64,
        gen:   Option<Generator>)  {
        
        todo!();
        /*
            bernoulli_kernel(self, p, check_generator<R>(gen));
        */
    }
    
    pub fn invoke(&mut self, 
        self_: &mut Tensor,
        p:     &Tensor,
        gen:   Option<Generator>)  {
        
        todo!();
        /*
            bernoulli_kernel(self, p_, check_generator<R>(gen));
        */
    }
}
