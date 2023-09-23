crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Distributions.h]

/**
  | ROCM hcc doesn't work well with using
  | in kernel functions
  |
  */
lazy_static!{
    /*
    #if defined(__CUDA_ARCH__)

    #define compat_exp compat::exp
    #define compat_ceil compat::ceil
    #define compat_floor compat::floor
    #define compat_log compat::log
    #define compat_pow compat::pow
    #define compat_sqrt compat::sqrt
    #define compat_tan compat::tan
    #define compat_abs compat::abs
    #define compat_log1p compat::log1p
    #elif defined(__HIPCC__)

    #define compat_exp hip::compat::exp
    #define compat_ceil hip::compat::ceil
    #define compat_floor hip::compat::floor
    #define compat_log hip::compat::log
    #define compat_pow hip::compat::pow
    #define compat_sqrt hip::compat::sqrt
    #define compat_tan hip::compat::tan
    #define compat_abs hip::compat::abs
    #define compat_log1p hip::compat::log1p
    #else
    #define compat_exp exp
    #define compat_ceil ceil
    #define compat_floor floor
    #define compat_log log
    #define compat_pow pow
    #define compat_sqrt sqrt
    #define compat_tan tan
    #define compat_abs abs
    #define compat_log1p log1p
    #endif
    */
}

/**
  | Here sampler_t should be function type
  | Scalar(void). For gpu "sampler" is a device
  | function, but since ROCM doesn't have
  | equivalent to nvstd::function, we use
  | a template type parameter to capture it.
  */
pub struct BaseSampler<Scalar,sampler_t> {
    sampler: Sampler,
}

impl BaseSampler<Scalar,sampler_t> {

    #[cfg(target_os = "cuda")] 
    pub fn new(sampler: &Sampler) -> Self {
    
        todo!();
        /*
        : sampler(sampler),

        
        */
    }

    #[cfg(target_os = "cuda")] 
    pub fn sample(&mut self) -> Scalar {
        
        todo!();
        /*
            return sampler();
        */
    }
}

/**
  | The function `sample_gamma` is is adapted from
  | Numpy's distributions.c implementation.
  |
  | It is MIT licensed, so here is the copyright:
  |
  | Copyright 2005 Robert Kern (robert.kern@gmail.com)
  | 
  | Permission is hereby granted, free
  | of charge, to any person obtaining a
  | copy of this software and associated
  | documentation files (the "Software"),
  | to deal in the Software without restriction,
  | including without limitation the rights
  | to use, copy, modify, merge, publish,
  | distribute, sublicense, and/or sell
  | copies of the Software, and to permit
  | persons to whom the Software is furnished
  | to do so, subject to the following conditions:
  | 
  | The above copyright notice and this
  | permission notice shall be included
  | in all copies or substantial portions
  | of the Software.
  | 
  | THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
  | WARRANTY OF ANY KIND, EXPRESS
  | 
  | OR IMPLIED, INCLUDING BUT NOT LIMITED
  | TO THE WARRANTIES OF
  | 
  | MERCHANTABILITY, FITNESS FOR A PARTICULAR
  | PURPOSE AND NONINFRINGEMENT.
  | 
  | IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
  | HOLDERS BE LIABLE FOR ANY
  | 
  | CLAIM, DAMAGES OR OTHER LIABILITY,
  | WHETHER IN AN ACTION OF CONTRACT,
  | 
  | TORT OR OTHERWISE, ARISING FROM, OUT
  | OF OR IN CONNECTION WITH THE
  | 
  | SOFTWARE OR THE USE OR OTHER DEALINGS
  | IN THE SOFTWARE.
  |
  */
#[cfg(target_os = "cuda")] 
pub fn sample_gamma<Scalar, accscalar_t, uniform_sampler_t, normal_sampler_t>(
        alpha:            Scalar,
        standard_uniform: &mut BaseSampler<AccScalar,UniformSampler>,
        standard_normal:  &mut BaseSampler<AccScalar,NormalSampler>) -> Scalar {

    todo!();
        /*
            accscalar_t scale = 1.0f;

      // Boost alpha for higher acceptance probability.
      if (alpha < 1.0f) {
        if (alpha == 0.f) return 0.f;
        scale *= compat_pow(1 - standard_uniform.sample(), 1.0f / alpha);
        alpha += 1.0f;
      }

      // This implements the acceptance-rejection method of Marsaglia and Tsang (2000)
      // doi:10.1145/358407.358414
      const accscalar_t d = alpha - 1.0f / 3.0f;
      const accscalar_t c = 1.0f / compat_sqrt(9.0f * d);
      for (;;) {
        accscalar_t x, y;
        do {
          x = standard_normal.sample();
          y = 1.0f + c * x;
        } while (y <= 0);
        const accscalar_t v = y * y * y;
        const accscalar_t u = 1 - standard_uniform.sample();
        const accscalar_t xx = x * x;
        if (u < 1.0f - 0.0331f * xx * xx)
          return static_cast<Scalar>(scale * d * v);
        if (compat_log(u) < 0.5f * xx + d * (1.0f - v + compat_log(v)))
          return static_cast<Scalar>(scale * d * v);
      }
        */
}

#[cfg(target_os = "cuda")] 
#[inline] pub fn polevl<Scalar>(
        x:   Scalar,
        A:   &[Scalar],
        len: usize) -> Scalar {

    todo!();
        /*
            Scalar result = 0;
      for (usize i = 0; i <= len; i++) {
        result = result * x + A[i];
      }
      return result;
        */
}

/**
  | the functions stirling_approx_tail,
  | binomial_inversion, and btrs are adapted
  | from TensorFlow's random_binomial_op.cc
  | implementation. That code is under
  | copyright: 2019 The TensorFlow Authors.
  | 
  | It was released under the Apache License,
  | Version 2.0 (the "License"), available
  | at: http://www.apache.org/licenses/LICENSE-2.0
  |
  */
#[cfg(target_os = "cuda")] 
pub fn stirling_approx_tail<Scalar>(k: Scalar) -> Scalar {

    todo!();
        /*
            const static Scalar kTailValues[] = {
        0.0810614667953272,
        0.0413406959554092,
        0.0276779256849983,
        0.02079067210376509,
        0.0166446911898211,
        0.0138761288230707,
        0.0118967099458917,
        0.0104112652619720,
        0.00925546218271273,
        0.00833056343336287
      };
      if (k <= 9) {
        return kTailValues[static_cast<usize>(k)];
      }
      Scalar kp1sq = (k + 1) * (k + 1);
      return (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
        */
}

#[cfg(target_os = "cuda")] 
pub fn binomial_inversion<Scalar, accscalar_t, uniform_sampler_t>(
        count:            Scalar,
        prob:             Scalar,
        standard_uniform: &mut BaseSampler<AccScalar,UniformSampler>) -> Scalar {

    todo!();
        /*
            accscalar_t U;
      accscalar_t geom_sum = 0;
      Scalar num_geom = 0;

      accscalar_t logprob = compat_log1p(-prob);

      while (1) {
        U = standard_uniform.sample();
        accscalar_t geom = compat_ceil(compat_log(U) / logprob);
        geom_sum += geom;
        if (geom_sum > count) {
          break;
        }
        num_geom = num_geom + 1;
      }
      return num_geom;
        */
}

#[cfg(target_os = "cuda")] 
pub fn btrs<Scalar, accscalar_t, uniform_sampler_t>(
        count:            Scalar,
        prob:             Scalar,
        standard_uniform: &mut BaseSampler<AccScalar,UniformSampler>) -> Scalar {

    todo!();
        /*
            Scalar k;
      accscalar_t U, V, us;

      // This is spq in the paper.
      const accscalar_t stddev = compat_sqrt(count * prob * (1 - prob));

      // Other coefficients for Transformed Rejection sampling.
      const accscalar_t b = 1.15 + 2.53 * stddev;
      const accscalar_t a = -0.0873 + 0.0248 * b + 0.01 * prob;
      const accscalar_t c = count * prob + 0.5;
      const accscalar_t v_r = 0.92 - 4.2 / b;
      const accscalar_t r = prob / (1 - prob);

      const accscalar_t alpha = (2.83 + 5.1 / b) * stddev;
      const accscalar_t m = compat_floor((count + 1) * prob);

      while (1) {
        U = standard_uniform.sample() - 0.5;
        V = standard_uniform.sample();

        us = 0.5 - compat_abs(U);
        k = static_cast<Scalar>(compat_floor((2 * a / us + b) * U + c));

        // Reject non-sensical answers.
        if (k < 0 || k > count) {
          continue;
        }
        // Region for which the box is tight, and we can return our calculated value.
        // This should happen 0.86 * v_r times. In the limit as n * p is large,
        // the acceptance rate converges to ~79% (and in the lower regime it is ~24%).
        if (us >= 0.07 && V <= v_r) {
          return k;
        }

        // This deviates from Hormann's BTRS algorithm, as there is a log missing.
        // For all (u, v) pairs outside of the bounding box, this calculates the
        // transformed-reject ratio.
        V = compat_log(V * alpha / (a / (us * us) + b));
        accscalar_t upperbound =
            ((m + 0.5) * compat_log((m + 1) / (r * (count - m + 1))) +
             (count + 1) * compat_log((count - m + 1) / (count - k + 1)) +
             (k + 0.5) * compat_log(r * (count - k + 1) / (k + 1)) +
             stirling_approx_tail<accscalar_t>(m) + stirling_approx_tail<accscalar_t>(count - m) -
             stirling_approx_tail<accscalar_t>(k) - stirling_approx_tail<accscalar_t>(count - k));

        if (V <= upperbound) {
          return k;
        }
      }
        */
}

#[cfg(target_os = "cuda")] 
pub fn sample_binomial<Scalar, accscalar_t, uniform_sampler_t>(
        count:            Scalar,
        prob:             Scalar,
        standard_uniform: &mut BaseSampler<AccScalar,UniformSampler>) -> Scalar {

    todo!();
        /*
            if (count <= 0.0 || prob <= 0.0) {
        return 0;
      } else if (prob >= 1.0) {
        return count;
      } else if (prob <= 0.5) {
        if (count * prob >= 10.0) {
          // btrs
          return btrs<Scalar, accscalar_t, uniform_sampler_t>(count, prob, standard_uniform);
        } else {
          // binomial inversion
          return binomial_inversion<Scalar, accscalar_t, uniform_sampler_t>(count, prob, standard_uniform);
        }
      } else if (prob > 0.5) {
        Scalar qprob = 1.0 - prob;
        if (count * qprob >= 10.0) {
          // btrs
          return count - btrs<Scalar, accscalar_t, uniform_sampler_t>(count, qprob, standard_uniform);
        } else {
          // count - binomial inversion
          return count - binomial_inversion<Scalar, accscalar_t, uniform_sampler_t>(count, qprob, standard_uniform);
        }
      } else {
        // prob is nan?
        return static_cast<Scalar>(NAN);
      }
        */
}



/**
  | This function is derived from the implementation
  | of the digamma function in the Cephes
  | Math Library.
  | 
  | See note [3-Clause BSD License for the
  | Cephes Math Library] in ATen/native/Math.h.
  |
  */
#[cfg(target_os = "cuda")] 
#[inline] pub fn digamma_one<Scalar, accscalar_t>(x: Scalar) -> Scalar {

    todo!();
        /*
            constexpr accscalar_t PSI_10 = 2.25175258906672110764;
      if (x == 0) {
        return INFINITY;
      }
      accscalar_t additional_summand = 0;
      int x_is_integer = x == compat_floor(x);
      if (x < 0) {
        if (x_is_integer) {
          return INFINITY;
        }
        // it is more standard to write this as recursion, but
        // nvcc does not like that
        additional_summand = -pi<Scalar> /
            compat_tan(pi<Scalar> * x);
        x = 1 - x;
      }

      // Push x to be >= 10
      accscalar_t result = 0;
      while (x < 10) {
        result -= 1 / x;
        x += 1;
      }
      if (x == 10) {
        return result + PSI_10 + additional_summand;
      }

      // Compute asymptotic digamma
      static const accscalar_t A[] = {
         8.33333333333333333333E-2,
        -2.10927960927960927961E-2,
         7.57575757575757575758E-3,
        -4.16666666666666666667E-3,
         3.96825396825396825397E-3,
        -8.33333333333333333333E-3,
         8.33333333333333333333E-2,
      };

      accscalar_t y = 0;
      if (x < 1.0e17f) {
        accscalar_t z = 1.0 / (x * x);
        y = z * polevl<accscalar_t>(z, A, 6);
      }
      return static_cast<Scalar>(
          result + compat_log(x) - (0.5f / x) - y + additional_summand);
        */
}



/**
  | Computes the reparameterized gradient
  | -(d/dalpha cdf(x;alpha)) / pdf(x;alpha) for
  | random number x drawn from a standard Gamma
  | distribution Gamma(alpha).
  |
  */
pub fn standard_gamma_grad_one<Scalar, accscalar_t>(
        alpha: Scalar,
        x:     Scalar) -> Scalar {

    todo!();
        /*
            // Use a Taylor series expansion for small x.
      accscalar_t x = static_cast<accscalar_t>(x_);
      accscalar_t alpha = static_cast<accscalar_t>(alpha_);
      if (x < 0.8f) {
        accscalar_t numer = 1;
        accscalar_t denom = alpha;
        auto series1 = numer / denom;
        auto series2 = numer / (denom * denom);
        for (int i = 1; i <= 5; ++i) {
          numer *= -x / static_cast<accscalar_t>(i);
          denom += 1;
          series1 += numer / denom;
          series2 += numer / (denom * denom);
        }
        const auto pow_x_alpha = compat_pow(x, alpha);
        const auto gamma_pdf = compat_pow(x, alpha - 1) * compat_exp(-x);
        const auto gamma_cdf = pow_x_alpha * series1;
        const auto gamma_cdf_alpha =
            (compat_log(x) - digamma_one<accscalar_t, accscalar_t>(alpha)) *
                gamma_cdf -
            pow_x_alpha * series2;
        const auto result = -gamma_cdf_alpha / gamma_pdf;
        return isnan(result) ? static_cast<Scalar>( 0.f ) : static_cast<Scalar>(result);
      }

      // Use a Rice saddle point expansion for large alpha.
      if (alpha > 8.0f) {
        if (0.9f * alpha <= x && x <= 1.1f * alpha) {
          const auto numer_1 = 1 + 24 * alpha * (1 + 12 * alpha);
          const auto numer_2 = 1440 * (alpha * alpha) + 6 * x * (53 - 120 * x)
              - 65 * x * x / alpha + alpha * (107 + 3600 * x);
          const auto denom = 1244160 * (alpha * alpha) * (alpha * alpha);
          return static_cast<Scalar>(numer_1 * numer_2 / denom);
        }
        const auto denom = compat_sqrt(8 * alpha);
        const auto term2 = denom / (alpha - x);
        const auto term3 = compat_pow(
            x - alpha - alpha * compat_log(x / alpha),
            static_cast<accscalar_t>(-1.5));
        const auto term23 = (x < alpha) ? term2 - term3 : term2 + term3;
        const auto term1 = compat_log(x / alpha) * term23 -
            compat_sqrt(2 / alpha) * (alpha + x) / ((alpha - x) * (alpha - x));
        const auto stirling = 1 + 1 / (12 * alpha) * (1 + 1 / (24 * alpha));
        const auto numer = x * term1;
        return static_cast<Scalar>(-stirling * numer / denom);
      }

      // Use a bivariate rational approximation to the reparameterized gradient.
      const auto u = compat_log(x / alpha);
      const auto v = compat_log(alpha);
      static const accscalar_t coef_uv[3][8] = {
        {0.16009398, -0.094634809, 0.025146376, -0.0030648343,
         1, 0.32668115, 0.10406089, 0.0014179084},
        {0.53487893, 0.1298071, 0.065735949, -0.0015649758,
         0.16639465, 0.020070113, -0.0035938915, -0.00058392623},
        {0.040121004, -0.0065914022, -0.0026286047, -0.0013441777,
         0.017050642, -0.0021309326, 0.00085092367, -1.5247877e-07},
      };
      accscalar_t coef_v[8];
      for (int i = 0; i < 8; ++ i) {
        coef_v[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
      }
      const auto p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
      const auto q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
      return static_cast<Scalar>(compat_exp(p / q));
        */
}



/**
  | Approximate reparameterized gradient of
  | Beta(x,alpha,beta) wrt alpha.
  |
  | Assumes x is close to zero and uses a Taylor
  | expansion.
  |
  */
#[cfg(target_os = "cuda")] 
#[inline] pub fn beta_grad_alpha_small<Scalar, accscalar_t>(
        x:     Scalar,
        alpha: Scalar,
        beta:  Scalar) -> Scalar {

    todo!();
        /*
            const Scalar factor = digamma_one<Scalar, accscalar_t>(alpha)
                            - digamma_one<Scalar, accscalar_t>(alpha + beta) - compat_log(x);
      Scalar numer = 1;
      Scalar series = numer / alpha * (factor + 1 / alpha);
      for (int i = 1; i <= 10; ++i) {
        Scalar casted_i = static_cast<Scalar>(i);
        numer *= (casted_i - beta) * x / casted_i;
        const Scalar denom = alpha + casted_i;
        series += numer / denom * (factor + 1 / denom);
      }
      const Scalar result = x * compat_pow(1 - x, -beta) * series;
      return isnan(result) ? static_cast<Scalar>( 0.f ) : result;
        */
}

/**
  | Approximate reparameterized gradient of
  | Beta(x,alpha,beta) wrt beta.
  |
  | Assumes x is close to zero and uses a Taylor
  | expansion.
  |
  */
#[cfg(target_os = "cuda")] 
#[inline] pub fn beta_grad_beta_small<Scalar, accscalar_t>(
        x:     Scalar,
        alpha: Scalar,
        beta:  Scalar) -> Scalar {

    todo!();
        /*
            const Scalar factor = digamma_one<Scalar, accscalar_t>(alpha + beta) - digamma_one<Scalar, accscalar_t>(beta);
      Scalar numer = 1, betas = 1, dbetas = 0, series = factor / alpha;
      for (int i = 1; i <= 8; ++i) {
        Scalar casted_i = static_cast<Scalar>(i);
        numer *= -x / casted_i;
        dbetas = dbetas * (beta - casted_i) + betas;
        betas = betas * (beta - casted_i);
        series += numer / (alpha + casted_i) * (dbetas + factor * betas);
      }
      const Scalar result = -compat_pow(1 - x, 1 - beta) * series;
      return isnan(result) ? static_cast<Scalar>( 0.f ) : result;
        */
}



/**
  | Approximate reparameterized gradient of
  | Beta(x,alpha,beta) wrt alpha.
  |
  | Assumes alpha and beta are both large and uses
  | a Rice saddle point expansion.
  |
  | To ensure numerical stability, this computation
  | is performed at higher precision.
  |
  */
#[cfg(target_os = "cuda")]
#[inline] pub fn beta_grad_alpha_mid<Scalar, accscalar_t>(
        x:     AccScalar,
        alpha: AccScalar,
        beta:  AccScalar) -> Scalar {

    todo!();
        /*
            const accscalar_t total = alpha + beta;
      const accscalar_t mean = alpha / total;
      const accscalar_t std = compat_sqrt(alpha * beta / (total + 1)) / total;
      if (mean - 0.1 * std <= x && x <= mean + 0.1 * std) {
        // Avoid the singularity at x = mean.
        const accscalar_t poly = 47 * x * (beta * beta) * (beta * beta) + alpha * (
                               (43 + 20 * (16 + 27 * beta) * x) * (beta * beta) * beta + alpha * (
                               3 * (59 + 180 * beta - 90 * x) * (beta * beta) + alpha * (
                               (453 + 1620 * beta * (1 - x) - 455 * x) * beta + alpha * (
                               8 * (1 - x) * (135 * beta - 11)))));
        const accscalar_t prefactor_num = (1 + 12 * alpha) * (1 + 12 * beta) / (total * total);
        const accscalar_t prefactor_den = 12960 * alpha * alpha * alpha * beta * beta * (1 + 12 * total);
        return prefactor_num / (1 - x) * poly / prefactor_den;
      }
      const accscalar_t prefactor = -x / compat_sqrt(2 * alpha * beta / total);
      const accscalar_t stirling = (1 + 1 / (12 * alpha) + 1 / (288 * alpha * alpha))
                                 * (1 + 1 / (12 * beta) + 1 / (288 * beta * beta))
                                 / (1 + 1 / (12 * total) + 1 / (288 * total * total));
      const accscalar_t term1_num = 2 * (alpha * alpha) * (x - 1) + alpha * beta * (x - 1) - x * (beta * beta);
      const accscalar_t axbx = alpha * (x - 1) + beta * x;
      const accscalar_t term1_den = compat_sqrt(2 * alpha / beta) * compat_pow(total, static_cast<accscalar_t>(1.5f)) * axbx * axbx;
      const accscalar_t term1 = term1_num / term1_den;
      const accscalar_t term2 = 0.5f * compat_log(alpha / (total * x));
      const accscalar_t term3_num = compat_sqrt(8 * alpha * beta / total);
      const accscalar_t term3_den = beta * x + alpha * (x - 1);
      const accscalar_t term3 = term3_num / term3_den;
      const accscalar_t term4_base = beta * compat_log(beta / (total * (1 - x))) +
                                   alpha * compat_log(alpha / (total * x));
      const accscalar_t term4 = compat_pow(term4_base, static_cast<accscalar_t>(-1.5f));
      const accscalar_t term1234 = term1 + term2 * (term3 + (x < mean ? term4 : -term4));
      return static_cast<Scalar>(stirling * prefactor * term1234);
        */
}

/**
  | Computes a scaled reparameterized gradient
  |
  |   -(d/dalpha cdf(x;alpha,beta)) / pdf(x;alpha,beta) / (1-x)
  |
  | for random number x drawn from a Beta
  | distribution Beta(alpha,beta).
  |
  | This function inputs total=alpha+beta to make
  | it easy to implement
  |
  | Dirichlet reparameterized gradients in terms of
  | Betas.
  |
  */
#[inline] pub fn dirichlet_grad_one<Scalar, accscalar_t>(
        x:     Scalar,
        alpha: Scalar,
        total: Scalar) -> Scalar {

    todo!();
        /*
            accscalar_t x_ = static_cast<accscalar_t>(x);
      accscalar_t alpha_ = static_cast<accscalar_t>(alpha);
      accscalar_t total_ = static_cast<accscalar_t>(total);

      const Scalar beta = total - alpha;
      const accscalar_t beta_ = total_ - alpha_;
      const Scalar boundary = total * x * (1 - x);

      // Use an asymptotic approximation for x close to 0.
      if (x <= 0.5f && boundary < 2.5f) {
        return _beta_grad_alpha_small<Scalar, accscalar_t>(x, alpha, beta);
      }

      // Use an asymptotic approximation for x close to 1.
      if (x >= 0.5f && boundary < 0.75f) {
        return -_beta_grad_beta_small<Scalar, accscalar_t>(1 - x, beta, alpha);
      }

      // Use an asymptotic approximation when alpha and (total - alpha) are both large.
      if (alpha > 6 && beta > 6) {
        return _beta_grad_alpha_mid<Scalar, accscalar_t>(x_, alpha_, beta_);
      }

      // Use a rational correction to an analytic approximation.
      static const accscalar_t c[2][3][3][4] = {
        {{{1.003668233, -0.01061107488, -0.0657888334, 0.01201642863},
          {0.6336835991, -0.3557432599, 0.05486251648, -0.001465281033},
          {-0.03276231906, 0.004474107445, 0.002429354597, -0.0001557569013}},
         {{0.221950385, -0.3187676331, 0.01799915743, 0.01074823814},
          {-0.2951249643, 0.06219954479, 0.01535556598, 0.001550077057},
          {0.02155310298, 0.004170831599, 0.001292462449, 6.976601077e-05}},
         {{-0.05980841433, 0.008441916499, 0.01085618172, 0.002319392565},
          {0.02911413504, 0.01400243777, -0.002721828457, 0.000751041181},
          {0.005900514878, -0.001936558688, -9.495446725e-06, 5.385558597e-05}}},
        {{{1, -0.02924021934, -0.04438342661, 0.007285809825},
          {0.6357567472, -0.3473456711, 0.05454656494, -0.002407477521},
          {-0.03301322327, 0.004845219414, 0.00231480583, -0.0002307248149}},
         {{0.5925320577, -0.1757678135, 0.01505928619, 0.000564515273},
          {0.1014815858, -0.06589186703, 0.01272886114, -0.0007316646956},
          {-0.007258481865, 0.001096195486, 0.0003934994223, -4.12701925e-05}},
         {{0.06469649321, -0.0236701437, 0.002902096474, -5.896963079e-05},
          {0.001925008108, -0.002869809258, 0.0008000589141, -6.063713228e-05},
          {-0.0003477407336, 6.959756487e-05, 1.097287507e-05, -1.650964693e-06}}},
      };
      const accscalar_t u = compat_log(x_);
      const accscalar_t a = compat_log(alpha_) - u;
      const accscalar_t b = compat_log(total_) - a;
      const accscalar_t pow_u[3] = {1, u, u * u};
      const accscalar_t pow_a[3] = {1, a, a * a};
      accscalar_t p = 0.0;
      accscalar_t q = 0.0;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          const accscalar_t ua = pow_u[i] * pow_a[j];
          p += ua * (c[0][i][j][0] + b * (c[0][i][j][1] + b * (c[0][i][j][2] + b * c[0][i][j][3])));
          q += ua * (c[1][i][j][0] + b * (c[1][i][j][1] + b * (c[1][i][j][2] + b * c[1][i][j][3])));
        }
      }
      const accscalar_t approx = x_ * (digamma_one<Scalar, accscalar_t>(total_) - digamma_one<Scalar, accscalar_t>(alpha_)) / beta_;
      return static_cast<Scalar>(p / q * approx);
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Distributions.cpp]


/*
  | This section is a counterpart to Distributions.cu
  |
  | The function `sample_poisson`
  |
  | is adapted from Numpy's distributions.c
  | implementation.
  |
  | It is MIT licensed, so here is the copyright:
  |
  | Copyright 2005 Robert Kern (robert.kern@gmail.com)
  | 
  | Permission is hereby granted, free
  | of charge, to any person obtaining a
  | copy of this software and associated
  | documentation files (the "Software"),
  | to deal in the Software without restriction,
  | including without limitation the rights
  | to use, copy, modify, merge, publish,
  | distribute, sublicense, and/or sell
  | copies of the Software, and to permit
  | persons to whom the Software is furnished
  | to do so, subject to the following conditions:
  | 
  | The above copyright notice and this
  | permission notice shall be included
  | in all copies or substantial portions
  | of the Software.
  | 
  | THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
  | WARRANTY OF ANY KIND, EXPRESS
  | 
  | OR IMPLIED, INCLUDING BUT NOT LIMITED
  | TO THE WARRANTIES OF
  | 
  | MERCHANTABILITY, FITNESS FOR A PARTICULAR
  | PURPOSE AND NONINFRINGEMENT.
  | 
  | IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
  | HOLDERS BE LIABLE FOR ANY
  | 
  | CLAIM, DAMAGES OR OTHER LIABILITY,
  | WHETHER IN AN ACTION OF CONTRACT,
  | 
  | TORT OR OTHERWISE, ARISING FROM, OUT
  | OF OR IN CONNECTION WITH THE
  | 
  | SOFTWARE OR THE USE OR OTHER DEALINGS
  | IN THE SOFTWARE.
  |
  */
pub fn sample_poisson(
        lambda:    f64,
        generator: *mut CPUGeneratorImpl) -> i64 {
    
    todo!();
        /*
            TORCH_CHECK(lambda >= 0, "invalid Poisson rate, expected rate to be non-negative");
      uniform_real_distribution<double> standard_uniform(0.0, 1.0);
      if (lambda >= 10) {
        // transformed rejection method, (Hoermann, 1993)
        i64 k;
        double U, V, a, b, invalpha, vr, us;

        double slam = sqrt(lambda);
        double loglam = log(lambda);
        b = 0.931 + 2.53 * slam;
        a = -0.059 + 0.02483 * b;
        invalpha = 1.1239 + 1.1328 / (b - 3.4);
        vr = 0.9277 - 3.6224 / (b - 2);

        while (1) {
          U = standard_uniform(generator) - 0.5;
          V = standard_uniform(generator);
          us = 0.5 - fabs(U);
          k = (i64)floor((2 * a / us + b) * U + lambda + 0.43);
          if ((us >= 0.07) && (V <= vr)) {
            return k;
          }
          if ((k < 0) || ((us < 0.013) && (V > us))) {
            continue;
          }
          if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <=
              (-lambda + k * loglam - lgamma((double)k + 1))) {
            return k;
          }
        }
      } else if (lambda == 0) {
        return 0;
      } else {
        i64 X;
        double prod, U, enlam;

        enlam = exp(-lambda);
        X = 0;
        prod = 1.0;
        while (1) {
          U = standard_uniform(generator);
          prod *= U;
          if (prod > enlam) {
            X += 1;
          } else {
            return X;
          }
        }
      }
        */
}

define_dispatch!{bernoulli_tensor_stub}
define_dispatch!{bernoulli_scalar_stub}
define_dispatch!{cauchy_stub}
define_dispatch!{exponential_stub}
define_dispatch!{multinomial_with_replacement_stub}
define_dispatch!{geometric_stub}
define_dispatch!{log_normal_stub}
define_dispatch!{uniform_stub}
define_dispatch!{normal_stub}
define_dispatch!{random_stub}
define_dispatch!{random_from_to_stub}
define_dispatch!{random_full_64_bits_range_stub}

// ==================================================== Bernoulli =====================================================

pub struct BernoulliStub<RNG> {

}

impl BernoulliStub<RNG> {

    pub fn invoke(&mut self, 
        self_: &mut Tensor,
        p:     &Tensor,
        gen:   Option<Generator>)  {
        
        todo!();
        /*
            bernoulli_tensor_stub(self.device().type(), self, p_, gen);
        */
    }
    
    pub fn invoke(&mut self, 
        self_: &mut Tensor,
        p:     f64,
        gen:   Option<Generator>)  {
        
        todo!();
        /*
            bernoulli_scalar_stub(self.device().type(), self, p, gen);
        */
    }
}

pub fn bernoulli_a(
        self_: &Tensor,
        gen:   Option<Generator>) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      result.bernoulli_(self, gen);
      return result;
        */
}

pub fn bernoulli_b(
        self_: &Tensor,
        p:     f64,
        gen:   Option<Generator>) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      result.bernoulli_(p, gen);
      return result;
        */
}


pub fn bernoulli_out<'a>(
        self_:  &Tensor,
        gen:    Option<Generator>,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::bernoulli_out_impl<BernoulliStub, Generator>(result, self, gen);
        */
}

pub fn bernoulli_c<'a>(
        self_: &mut Tensor,
        p:     &Tensor,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::bernoulli_impl_<BernoulliStub, Generator>(self, p_, gen);
        */
}

pub fn bernoulli_d<'a>(
        self_: &mut Tensor,
        p:     f64,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::bernoulli_impl_<BernoulliStub, Generator>(self, p, gen);
        */
}

// ================================================== LogNormal =======================================================

pub struct LogNormalStub<RNG> {

}

impl LogNormalStub<RNG> {
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        mean: f64,
        std:  f64,
        gen:  Option<Generator>)  {
        
        todo!();
        /*
            log_normal_stub(iter.device_type(), iter, mean, std, gen);
        */
    }
}

pub fn log_normal<'a>(
        self_: &mut Tensor,
        mean:  f64,
        std:   f64,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::log_normal_impl_<LogNormalStub, Generator>(self, mean, std, gen);
        */
}

// ==================================================== Cauchy ========================================================

pub struct CauchyStub<RNG> {

}

impl CauchyStub<RNG> {
    
    pub fn invoke(&mut self, 
        iter:   &mut TensorIteratorBase,
        median: f64,
        sigma:  f64,
        gen:    Option<Generator>)  {
        
        todo!();
        /*
            cauchy_stub(iter.device_type(), iter, median, sigma, gen);
        */
    }
}

pub fn cauchy<'a>(
        self_:  &mut Tensor,
        median: f64,
        sigma:  f64,
        gen:    Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::cauchy_impl_<CauchyStub, Generator>(self, median, sigma, gen);
        */
}

// ================================================== Exponential =====================================================

pub struct ExponentialStub<RNG> {

}

impl ExponentialStub<RNG> {
    
    pub fn invoke(&mut self, 
        iter:   &mut TensorIteratorBase,
        lambda: f64,
        gen:    Option<Generator>)  {
        
        todo!();
        /*
            exponential_stub(iter.device_type(), iter, lambda, gen);
        */
    }
}

pub fn exponential<'a>(
        self_:  &mut Tensor,
        lambda: f64,
        gen:    Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::exponential_impl_<ExponentialStub, Generator>(self, lambda, gen);
        */
}



// =================================================== Geometric ======================================================

pub struct GeometricStub<RNG> {

}

impl GeometricStub<RNG> {
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        p:    f64,
        gen:  Option<Generator>)  {
        
        todo!();
        /*
            geometric_stub(iter.device_type(), iter, p, gen);
        */
    }
}

pub fn geometric<'a>(
        self_: &mut Tensor,
        p:     f64,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::geometric_impl_<GeometricStub, Generator>(self, p, gen);
        */
}

// ==================================================== Uniform =======================================================

pub struct UniformStub<RNG> {

}

impl UniformStub<RNG> {
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        from: f64,
        to:   f64,
        gen:  Option<Generator>)  {
        
        todo!();
        /*
            uniform_stub(iter.device_type(), iter, from, to, gen);
        */
    }
}

pub struct UniformMeta<RNG> {

}

impl UniformMeta<RNG> {

    // No-op!
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        from: f64,
        to:   f64,
        gen:  Option<Generator>)  {
        
        todo!();
        /*
        
        */
    }
}

pub fn uniform<'a>(
        self_: &mut Tensor,
        from:  f64,
        to:    f64,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::uniform_impl_<UniformStub, Generator>(self, from, to, gen);
        */
}

pub fn uniform_meta<'a>(
        self_: &mut Tensor,
        from:  f64,
        to:    f64,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::uniform_impl_<UniformMeta, Generator>(self, from, to, gen);
        */
}

// ==================================================== Normal ========================================================

pub struct NormalStub<RNG> {

}

impl NormalStub<RNG> {
    
    pub fn invoke(&mut self, 
        self_: &mut Tensor,
        mean:  f64,
        std:   f64,
        gen:   Option<Generator>)  {
        
        todo!();
        /*
            normal_stub(self.device().type(), self, mean, std, gen);
        */
    }
}

pub fn normal_a<'a>(
        self_: &mut Tensor,
        mean:  f64,
        std:   f64,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::normal_impl_<NormalStub, Generator>(self, mean, std, gen);
        */
}

pub fn normal_meta<'a>(
        self_: &mut Tensor,
        mean:  f64,
        std:   f64,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);  // TODO: dedupe
      return self;
        */
}

pub fn normal_out_a<'a>(
        mean:   &Tensor,
        std:    f64,
        gen:    Option<Generator>,
        output: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, gen);
        */
}

pub fn normal_out_b<'a>(
        mean:   f64,
        std:    &Tensor,
        gen:    Option<Generator>,
        output: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, gen);
        */
}

pub fn normal_out_tensor_tensor<'a>(
        mean:   &Tensor,
        std:    &Tensor,
        gen:    Option<Generator>,
        output: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, gen);
        */
}

pub fn normal_b(
        mean: &Tensor,
        std:  f64,
        gen:  Option<Generator>) -> Tensor {
    
    todo!();
        /*
            return native::templates::normal_impl<NormalStub, Generator>(mean, std, gen);
        */
}

pub fn normal_c(
        mean: f64,
        std:  &Tensor,
        gen:  Option<Generator>) -> Tensor {
    
    todo!();
        /*
            return native::templates::normal_impl<NormalStub, Generator>(mean, std, gen);
        */
}

pub fn normal_d(
        mean: &Tensor,
        std:  &Tensor,
        gen:  Option<Generator>) -> Tensor {
    
    todo!();
        /*
            return native::templates::normal_impl<NormalStub, Generator>(mean, std, gen);
        */
}



// ==================================================== Random ========================================================

pub struct RandomStub<RNG> {

}

impl RandomStub<RNG> {
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        gen:  Option<Generator>)  {
        
        todo!();
        /*
            random_stub(iter.device_type(), iter, gen);
        */
    }
}

pub fn random<'a>(
    self_: &mut Tensor,
    gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::random_impl<RandomStub, Generator>(self, gen);
        */
}

pub struct RandomFromToStub<RNG> {

}

impl RandomFromToStub<RNG> {
    
    pub fn invoke(&mut self, 
        iter:  &mut TensorIteratorBase,
        range: u64,
        from:  i64,
        gen:   Option<Generator>)  {
        
        todo!();
        /*
            random_from_to_stub(iter.device_type(), iter, range, from, gen);
        */
    }
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        gen:  Option<Generator>)  {
        
        todo!();
        /*
            random_full_64_bits_range_stub(iter.device_type(), iter, gen);
        */
    }
}

pub struct RandomFromToMeta<RNG> {

}

impl RandomFromToMeta<RNG> {
    
    pub fn invoke(&mut self, 
        iter:  &mut TensorIteratorBase,
        range: u64,
        from:  i64,
        gen:   Option<Generator>)  {
        
        todo!();
        /*
            // No-op!
        */
    }
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        gen:  Option<Generator>)  {
        
        todo!();
        /*
        
        */
    }
}

pub fn random_range<'a>(
        self_: &mut Tensor,
        from:  i64,
        to:    Option<i64>,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::random_from_to_impl<RandomFromToStub, Generator>(self, from, to, gen);
        */
}

pub fn random_to<'a>(
        self_: &mut Tensor,
        to:    i64,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return random_(self, 0, to, gen);
        */
}

pub fn random_meta<'a>(
        self_: &mut Tensor,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            // No error checking yay
      return self;
        */
}

pub fn random_meta_range<'a>(
        self_: &mut Tensor,
        from:  i64,
        to:    Option<i64>,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::templates::random_from_to_impl<RandomFromToMeta, Generator>(self, from, to, gen);
        */
}

pub fn random_meta_until<'a>(
        self_: &mut Tensor,
        to:    i64,
        gen:   Option<Generator>) -> &'a mut Tensor {
    
    todo!();
        /*
            return random_meta_(self, 0, to, gen);
        */
}

// ====================================================================================================================

pub fn standard_gamma_grad_cpu(
        self_:  &Tensor,
        output: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor ret = empty(self.sizes(), self.options());
      auto iter = TensorIteratorConfig()
        .add_output(ret)
        .add_input(self)
        .add_input(output)
        .build();
      AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "_standard_gamma_grad_cpu", [&] {
        cpu_serial_kernel(iter, [](Scalar self_val, Scalar output_val) -> Scalar{
          return standard_gamma_grad_one<Scalar, double>(self_val, output_val);
        });
      });
      return ret;
        */
}

pub fn dirichlet_grad_cpu(
        x:     &Tensor,
        alpha: &Tensor,
        total: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor ret = empty(x.sizes(), x.options());
      auto iter = TensorIteratorConfig()
        .add_output(ret)
        .add_input(x)
        .add_input(alpha)
        .add_input(total)
        .build();
      AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "_dirichlet_grad_cpu", [&] {
        cpu_serial_kernel(iter, [](Scalar x_val, Scalar alpha_val, Scalar total_val) -> Scalar{
          return dirichlet_grad_one<Scalar, double>(x_val, alpha_val, total_val);
        });
      });
      return ret;
        */
}

/**
  | This section is a counterpart to Distributions.cu
  |
  */
pub fn s_binomial_cpu(
        count: &Tensor,
        prob:  &Tensor,
        gen:   Option<Generator>) -> Tensor {
    
    todo!();
        /*
            Tensor ret = zeros(count.sizes(), count.options());
      auto iter = TensorIteratorConfig()
        .add_output(ret)
        .add_input(count)
        .add_input(prob)
        .build();
      AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "binomial_cpu", [&] {
        CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
        // See Note [Acquire lock when using random generators]
        lock_guard<mutex> lock(generator->mutex_);
        cpu_serial_kernel(iter, [generator](Scalar count_val, Scalar prob_val) -> Scalar{
          auto uniform_lambda = [generator] () {
            uniform_real_distribution<double> standard_uniform(0.0, 1.0);
            return standard_uniform(generator);
          };
          BaseSampler<double, decltype(uniform_lambda)> standard_uniform(uniform_lambda);

          auto sample = sample_binomial<Scalar, double, decltype(uniform_lambda)>(count_val, prob_val, standard_uniform);
          return static_cast<Scalar>(sample);
        });
      });
      return ret;
        */
}

pub fn s_poisson_cpu(
        lambda: &Tensor,
        gen:    Option<Generator>) -> Tensor {
    
    todo!();
        /*
            Tensor ret = zeros(lambda.sizes(), lambda.options());
      auto iter = TensorIteratorConfig()
        .add_output(ret)
        .add_input(lambda)
        .build();
      AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "poisson_cpu", [&] {
        CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
        // See Note [Acquire lock when using random generators]
        lock_guard<mutex> lock(generator->mutex_);
        cpu_serial_kernel(iter, [generator](Scalar lambda_val) -> Scalar{
          return static_cast<Scalar>(sample_poisson(static_cast<double>(lambda_val), generator));
        });
      });
      return ret;
        */
}

pub fn s_gamma_cpu(
        alpha: &Tensor,
        gen:   Option<Generator>) -> Tensor {
    
    todo!();
        /*
            Tensor ret = zeros(alpha.sizes(), alpha.options());
      auto iter = TensorIteratorConfig()
        .add_output(ret)
        .add_input(alpha)
        .build();
      AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "gamma_cpu", [&] {
        CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
        // See Note [Acquire lock when using random generators]
        lock_guard<mutex> lock(generator->mutex_);
        cpu_serial_kernel(iter, [generator](Scalar alpha_val) -> Scalar{
          auto uniform_lambda = [generator] () {
            uniform_real_distribution<double> standard_uniform(0.0, 1.0);
            return standard_uniform(generator);
          };
          BaseSampler<double, decltype(uniform_lambda)> standard_uniform(uniform_lambda);

          auto normal_lambda = [generator] () {
            normal_distribution<double> normal(0.0, 1.0);
            return normal(generator);
          };
          BaseSampler<double, decltype(normal_lambda)> standard_normal(normal_lambda);
          auto sample = sample_gamma<Scalar, double, decltype(uniform_lambda), decltype(normal_lambda)>(alpha_val, standard_uniform, standard_normal);
          return max(Scalar::min, (Scalar) sample);
        });
      });

      return ret;
        */
}

pub fn s_dirichlet_cpu(
        alpha: &Tensor,
        gen:   Option<Generator>) -> Tensor {
    
    todo!();
        /*
            Tensor ret = zeros(alpha.sizes(), alpha.options());
      AT_DISPATCH_FLOATING_TYPES(ret.scalar_type(), "dirichlet", [&] {
        Tensor gamma = zeros(alpha.sizes(), alpha.options().dtype(ScalarType::Double));
        CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultCPUGenerator());
        // See Note [Acquire lock when using random generators]
        lock_guard<mutex> lock(generator->mutex_);
        /* Generate gamma sample by casting alpha to double to prevent underflow. */
        auto iter1 = TensorIteratorConfig()
          .add_output(gamma)
          .add_input(alpha)
          .check_all_same_dtype(false)
          .build();
        cpu_serial_kernel(iter1, [generator](Scalar alpha_val) -> double{
          auto uniform_lambda = [generator] () {
            uniform_real_distribution<double> standard_uniform(0.0, 1.0);
            return standard_uniform(generator);
          };
          BaseSampler<double, decltype(uniform_lambda)> standard_uniform(uniform_lambda);

          auto normal_lambda = [generator] () {
            normal_distribution<double> normal(0.0, 1.0);
            return normal(generator);
          };
          BaseSampler<double, decltype(normal_lambda)> standard_normal(normal_lambda);
          auto sample = sample_gamma<double, double, decltype(uniform_lambda), decltype(normal_lambda)>
            (alpha_val, standard_uniform, standard_normal);
          return max(double::min, sample);
        });
        /* Normalize and cast back to Scalar. */
        Tensor gamma_sum = gamma.sum(-1, true).expand(alpha.sizes());
        auto iter2 = TensorIteratorConfig()
          .add_output(ret)
          .add_input(gamma)
          .add_input(gamma_sum)
          .check_all_same_dtype(false)
          .build();
        cpu_serial_kernel(iter2, [](double gamma_val, double gamma_sum_val) -> Scalar{
          auto ret_val = gamma_val / gamma_sum_val;
          auto min_val = Scalar::min;
          auto max_val = nexttoward(static_cast<Scalar>(1.0f), 0.0f);
          return min(max_val, max(min_val, static_cast<Scalar>(ret_val)));
        });
      });
      return ret;
        */
}

/* The largest consecutive integer representable in float32 (2^24) */
pub const FLOAT32_MAX_CONSECUTIVE_INT: i64 = 1 << (FLT_MANT_DIG);

pub fn multinomial_out<'a>(
        self_:            &Tensor,
        n_sample:         i64,
        with_replacement: bool,
        gen:              Option<Generator>,
        result:           &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          result.device() == self.device(),
          "multinomial arguments must have the same device");
      TORCH_CHECK(
          self.dim() > 0 && self.dim() <= 2, "prob_dist must be 1 or 2 dim");
      TORCH_CHECK(
          isFloatingType(self.scalar_type()),
          "multinomial only supports floating-point dtypes for input, got: ",
          self.scalar_type());
      TORCH_CHECK(result.scalar_type() == ScalarType::Long,
          "multinomial expects Long tensor out, got: ", result.scalar_type());
      TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
      i64 n_categories = self.size(-1);
      TORCH_CHECK(with_replacement || (n_sample <= n_categories),
          "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
      // Since the index tensor is float, numCategories cannot exceed max
      // float integer precision
      TORCH_CHECK(
          n_categories <= FLOAT32_MAX_CONSECUTIVE_INT,
          "number of categories cannot exceed 2^24");

      if (self.dim() == 1) {
        result.resize_({n_sample});
      } else {
        const i64 n_dist = self.size(0);
        result.resize_({n_dist, n_sample});
      }
      if (result.numel() == 0) {
        return result;
      }

      // Fast-path for no replacement.
      // Reference:
      // https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
      // Half is not supported on CPU.
      TORCH_CHECK(
          !(self.device().is_cpu() && self.scalar_type() == ScalarType::Half),
          "multinomial is not implemented for half on CPU");
      if (!with_replacement) {
        // Sanity checks on `self`.
        auto is_valid = ((self.max() < INFINITY) & (self.min() >= 0)).item();
        TORCH_CHECK(
            is_valid.to<bool>(),
            "probability tensor contains either `inf`, `nan` or element < 0");
        bool zero_prob_condition;
        if (self.dim() == 1){
          zero_prob_condition = (self.sum() == 0).item().to<bool>();
        } else {
          zero_prob_condition = (self.sum(1) == 0).sum().item().to<bool>();
        }
        TORCH_CHECK(
            !zero_prob_condition,
            "invalid multinomial distribution (sum of probabilities <= 0)");

        // The algorithm is from gumbel softmax.
        // s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1)
        // Here we can apply exp to the formula which will not affect result of
        // argmax or topk. Then we have
        // s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
        // We can also simplify the formula above by
        // s = argmax( p / q ) where q ~ Exp(1)
        Tensor q = empty_like(self).exponential_(1, gen);
        // In theory the probability to generate 0 from exponential distribution is
        // 0. However, on CUDA side there is a protection to avoid 0s, but on CPU
        // side, there is a very low probability to generate 0 from
        // exponential<double>. The probability is about 2^(-DBL_MANT_DIG). We just
        // ignore it here, but there may be some risk to get invalid output on CPU.
        div_out(q, self, q);
        if (n_sample == 1) {
          argmax_out(result, q, /*dim=*/-1, /*keepdim=*/true);
        } else {
          Tensor vals = empty(result.sizes(), self.options());
          topk_out(vals, result, q, n_sample);
        }
        return result;
      }

      multinomial_with_replacement_stub(
          result.device().type(), result, self, n_sample, gen);
      return result;
        */
}

pub fn multinomial(
        self_:            &Tensor,
        n_sample:         i64,
        with_replacement: bool,
        gen:              Option<Generator>) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options().dtype(kLong));
      native::multinomial_out(self, n_sample, with_replacement, gen, result);
      return result;
        */
}
