crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/half_test.cpp]

#[test] fn test_half_arithmetic() {
    todo!();
    /*
    
      Half zero = 0;
      Half one = 1;
      ASSERT_EQ(zero + one, one);
      ASSERT_EQ(zero + zero, zero);
      ASSERT_EQ(zero * one, zero);
      ASSERT_EQ(one * one, one);
      ASSERT_EQ(one / one, one);
      ASSERT_EQ(one - one, zero);
      ASSERT_EQ(one - zero, one);
      ASSERT_EQ(zero - one, -one);
      ASSERT_EQ(one + one, Half(2));
      ASSERT_EQ(one + one, 2);

    */
}

#[test] fn test_half_comparisions() {
    todo!();
    /*
    
      Half zero = 0;
      Half one = 1;
      ASSERT_LT(zero, one);
      ASSERT_LT(zero, 1);
      ASSERT_GT(1, zero);
      ASSERT_GE(0, zero);
      ASSERT_NE(0, one);
      ASSERT_EQ(zero, 0);
      ASSERT_EQ(zero, zero);
      ASSERT_EQ(zero, -zero);

    */
}

#[test] fn test_half_cast() {
    todo!();
    /*
    
      Half value = 1.5f;
      ASSERT_EQ((int)value, 1);
      ASSERT_EQ((short)value, 1);
      ASSERT_EQ((long long)value, 1LL);
      ASSERT_EQ((float)value, 1.5f);
      ASSERT_EQ((double)value, 1.5);
      ASSERT_EQ((bool)value, true);
      ASSERT_EQ((bool)Half(0.0f), false);

    */
}

#[test] fn test_half_construction() {
    todo!();
    /*
    
      ASSERT_EQ(Half((short)3), Half(3.0f));
      ASSERT_EQ(Half((unsigned short)3), Half(3.0f));
      ASSERT_EQ(Half(3), Half(3.0f));
      ASSERT_EQ(Half(3U), Half(3.0f));
      ASSERT_EQ(Half(3LL), Half(3.0f));
      ASSERT_EQ(Half(3ULL), Half(3.0f));
      ASSERT_EQ(Half(3.5), Half(3.5f));

    */
}

pub fn to_string(h: &f16) -> String {
    
    todo!();
        /*
            stringstream ss;
      ss << h;
      return ss.str();
        */
}

#[test] fn test_half_2string() {
    todo!();
    /*
    
      ASSERT_EQ(to_string(Half(3.5f)), "3.5");
      ASSERT_EQ(to_string(Half(-100.0f)), "-100");

    */
}

#[test] fn test_half_numeric_limits() {
    todo!();
    /*
    
      using limits = numeric_limits<Half>;
      ASSERT_EQ(limits::lowest(), -65504.0f);
      ASSERT_EQ(limits::max(), 65504.0f);
      ASSERT_GT(limits::min(), 0);
      ASSERT_LT(limits::min(), 1);
      ASSERT_GT(limits::denorm_min(), 0);
      ASSERT_EQ(limits::denorm_min() / 2, 0);
      ASSERT_EQ(limits::infinity(), numeric_limits<float>::infinity());
      ASSERT_NE(limits::quiet_NaN(), limits::quiet_NaN());
      ASSERT_NE(limits::signaling_NaN(), limits::signaling_NaN());

    */
}

/**
  | Check the declared type of members of
  | numeric_limits<Half> matches the
  | declared type of that member on numeric_limits<float>
  |
  */
#[macro_export] macro_rules! assert_same_type {
    ($name:ident) => {
        /*
        
          static_assert(                                              
              is_same<                                           
                  decltype(numeric_limits<Half>::name),          
                  decltype(numeric_limits<float>::name)>::value, 
              "decltype(" #name ") differs")
        */
    }
}

assert_same_type!(is_specialized);
assert_same_type!(is_signed);
assert_same_type!(is_integer);
assert_same_type!(is_exact);
assert_same_type!(has_infinity);
assert_same_type!(has_quiet_NaN);
assert_same_type!(has_signaling_NaN);
assert_same_type!(has_denorm);
assert_same_type!(has_denorm_loss);
assert_same_type!(round_style);
assert_same_type!(is_iec559);
assert_same_type!(is_bounded);
assert_same_type!(is_modulo);
assert_same_type!(digits);
assert_same_type!(digits10);
assert_same_type!(max_digits10);
assert_same_type!(radix);
assert_same_type!(min_exponent);
assert_same_type!(min_exponent10);
assert_same_type!(max_exponent);
assert_same_type!(max_exponent10);
assert_same_type!(traps);
assert_same_type!(tinyness_before);

#[test] fn test_half_common_math() {
    todo!();
    /*
    
      float threshold = 0.00001;
      assert(abs(lgamma(Half(10.0)) - lgamma(10.0f)) <= threshold);
      assert(abs(exp(Half(1.0)) - exp(1.0f)) <= threshold);
      assert(abs(log(Half(1.0)) - log(1.0f)) <= threshold);
      assert(abs(log10(Half(1000.0)) - log10(1000.0f)) <= threshold);
      assert(abs(log1p(Half(0.0)) - log1p(0.0f)) <= threshold);
      assert(abs(log2(Half(1000.0)) - log2(1000.0f)) <= threshold);
      assert(abs(expm1(Half(1.0)) - expm1(1.0f)) <= threshold);
      assert(abs(cos(Half(0.0)) - cos(0.0f)) <= threshold);
      assert(abs(sin(Half(0.0)) - sin(0.0f)) <= threshold);
      assert(abs(sqrt(Half(100.0)) - sqrt(100.0f)) <= threshold);
      assert(abs(ceil(Half(2.4)) - ceil(2.4f)) <= threshold);
      assert(abs(floor(Half(2.7)) - floor(2.7f)) <= threshold);
      assert(abs(trunc(Half(2.7)) - trunc(2.7f)) <= threshold);
      assert(abs(acos(Half(-1.0)) - acos(-1.0f)) <= threshold);
      assert(abs(cosh(Half(1.0)) - cosh(1.0f)) <= threshold);
      assert(abs(acosh(Half(1.0)) - acosh(1.0f)) <= threshold);
      assert(abs(asin(Half(1.0)) - asin(1.0f)) <= threshold);
      assert(abs(sinh(Half(1.0)) - sinh(1.0f)) <= threshold);
      assert(abs(asinh(Half(1.0)) - asinh(1.0f)) <= threshold);
      assert(abs(tan(Half(0.0)) - tan(0.0f)) <= threshold);
      assert(abs(atan(Half(1.0)) - atan(1.0f)) <= threshold);
      assert(abs(tanh(Half(1.0)) - tanh(1.0f)) <= threshold);
      assert(abs(erf(Half(10.0)) - erf(10.0f)) <= threshold);
      assert(abs(erfc(Half(10.0)) - erfc(10.0f)) <= threshold);
      assert(abs(abs(Half(-3.0)) - abs(-3.0f)) <= threshold);
      assert(abs(round(Half(2.3)) - round(2.3f)) <= threshold);
      assert(
          abs(pow(Half(2.0), Half(10.0)) - pow(2.0f, 10.0f)) <=
          threshold);
      assert(
          abs(atan2(Half(7.0), Half(0.0)) - atan2(7.0f, 0.0f)) <=
          threshold);
    #ifdef __APPLE__
      // @TODO: can macos do implicit conversion of Half?
      assert(
          abs(isnan(static_cast<float>(Half(0.0))) - isnan(0.0f)) <=
          threshold);
      assert(
          abs(isinf(static_cast<float>(Half(0.0))) - isinf(0.0f)) <=
          threshold);
    #else
      assert(abs(isnan(Half(0.0)) - isnan(0.0f)) <= threshold);
      assert(abs(isinf(Half(0.0)) - isinf(0.0f)) <= threshold);
    #endif

    */
}
