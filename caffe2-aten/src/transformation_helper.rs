// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/TransformationHelper.h]

lazy_static!{
    /*
    // Using DistAccumType in accumulate types for distributions.
    // Note: Ideally we'd be using ATen/AccumulateType.h but looks
    // like the there is some inconsistency in how accumulate types
    // are mapped currently, e.g. for the cpu side, float is mapped
    // to double.
    template <typename T>
    struct DistAccumType {  };

    #if defined(__CUDACC__) || defined(__HIPCC__)
    template <> struct DistAccumType<half> { using type = float; };
    #endif
    template <> struct DistAccumType<BFloat16> { using type = float; };
    template <> struct DistAccumType<Half> { using type = float; };
    template <> struct DistAccumType<float> { using type = float; };
    template <> struct DistAccumType<double> { using type = double; };

    template <typename T>
    using dist_acctype = typename DistAccumType<T>::type;


    /**
     * A transformation function for `torch.Tensor.random_()`, when both `from` and `to` are specified.
     * `range` is `to - from`
     * `base` is `from`
     */
    template <typename T, typename V>
     inline T uniform_int_from_to(V val, u64 range, i64 base) {
      return static_cast<T>(static_cast<i64>((val % range) + base));
    }

    /**
     * A transformation function for `torch.Tensor.random_()`, when `from=min_value(i64)` and to=None
     */
    template <typename T, typename V>
     inline T uniform_int_full_range(V val) {
      return static_cast<T>(static_cast<i64>(val));
    }

    /**
     * A transformation function for `torch.Tensor.random_()`, when used without specifying `from` and `to`.
     * In order to prevent compiler warnings reported in GitHub issue 46391, T can't be float or double
     * in this overloaded version
     */
    template <typename T, typename V>
     inline typename enable_if<!(is_floating_point<T>::value), T>::type uniform_int(V val) {
      if (is_same<T, bool>::value) {
        return static_cast<bool>(val & 1);
      } else if (is_same<T, i64>::value) {
        return static_cast<T>(val % (static_cast<u64>(T::max) + 1));
      } else if (is_same<T, Half>::value || is_same<T, BFloat16>::value) {
        return static_cast<T>(val % static_cast<u64>((1ULL << numeric_limits<T>::digits) + 1));
      } else if (is_integral<T>::value) {
        return static_cast<T>(val % (static_cast<u64>(T::max) + 1));
      } else {
        assert(false);
        return 0;
      }
    }

    /**
     * An overloaded transformation function for `torch.Tensor.random_()`, when used without specifying `from` and `to`,
     * added to fix compiler warnings reported in GitHub issue 46391. T is either float or double in this version.
     */
    template<typename T, typename V>
     inline 
     typename enable_if<is_floating_point<T>::value, T>::type 
     uniform_int(V val) {
      return static_cast<T>(val % static_cast<u64>((1ULL << numeric_limits<T>::digits) + 1));
    }
    */
}


#[inline] pub fn uniform_real<T, V>(
    val:  V,
    from: T,
    to:   T) -> DistAccType<T> {

    todo!();
        /*
            constexpr auto MASK = static_cast<V>((static_cast<u64>(1) << numeric_limits<T>::digits) - 1);
      constexpr auto DIVISOR = static_cast<dist_acctype<T>>(1) / (static_cast<u64>(1) << numeric_limits<T>::digits);
      dist_acctype<T> x = (val & MASK) * DIVISOR;
      return (x * (to - from) + from);
        */
}

/**
  | Transforms normally distributed `val`
  | with mean 0.0 and standard deviation
  | 1.0 to normally distributed with `mean`
  | and standard deviation `std`.
  |
  */
#[inline] pub fn normal<T>(
        val:  T,
        mean: T,
        std:  T) -> T {

    todo!();
        /*
            return val * std + mean;
        */
}

/**
  | Transforms uniformly distributed
  | `val` between 0.0 and 1.0 to
  | Cauchy distribution with location
  | parameter `median` and scale parameter
  | `sigma`.
  |
  */
#[inline] pub fn cauchy<T>(
        val:    T,
        median: T,
        sigma:  T) -> T {

    todo!();
        /*
            // https://en.wikipedia.org/wiki/Cauchy_distribution#Cumulative_distribution_function
      return median + sigma * tan(pi<T> * (val - static_cast<T>(0.5)));
        */
}


/**
  | Transforms uniformly distributed
  | `val` between 0.0 and 1.0 to exponentialy
  | distributed with `lambda` parameter
  | of the distribution.
  |
  */
#[inline] pub fn exponential<T>(
    val:    T,
    lambda: T) -> T {

    //#[__ubsan_ignore_float_divide_by_zero__] 

    todo!();
        /*
            // https://en.wikipedia.org/wiki/Exponential_distribution#Generating_exponential_variates
      // Different implementations for CUDA and CPU to preserve original logic
      // TODO: must be investigated and unified!!!
      // https://github.com/pytorch/pytorch/issues/38662
    #if defined(__CUDACC__) || defined(__HIPCC__)
          // BEFORE TOUCHING THIS CODE READ: https://github.com/pytorch/pytorch/issues/16706
          // curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
          // we need log to be not 0, and not underflow when converted to half
          // fast __logf approximation can underflow, so set log to -epsilon/2 for 1 or close to 1 args
      auto log = val >= static_cast<T>(1.) - numeric_limits<T>::epsilon() / 2
          ? -numeric_limits<T>::epsilon() / 2
          : log(val);
      return static_cast<T>(-1.0) / lambda * log;
    #else
      return static_cast<T>(-1.0) / lambda * log(static_cast<T>(1.0) - val);
    #endif
        */
}

/**
  | Transforms uniformly distributed
  | `val` between 0.0 and 1.0 to geometricaly
  | distributed with success probability `p`.
  |
  */
#[inline] pub fn geometric<T>(val: T, p: T) -> T {

    todo!();
        /*
            // https://en.wikipedia.org/wiki/Geometric_distribution#Related_distributions
      return static_cast<T>(::ceil(log(val) / log(static_cast<T>(1.0) - p)));
        */
}

/**
  | Transforms normally distributed `val`
  | to log-normally distributed.
  |
  */
#[inline] pub fn log_normal<T>(val: T) -> T {

    todo!();
        /*
            // https://en.wikipedia.org/wiki/Log-normal_distribution#Mode,_median,_quantiles
      return exp(val);
        */
}

/**
  | Transforms uniformly distributed
  | `val` between 0.0 and 1.0 to bernoulli
  | distributed with success probability `p`.
  |
  */
#[inline] pub fn bernoulli<T>(val: T, p: T) -> T {

    todo!();
        /*
            return val < p;
        */
}
