// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/pow_test.cpp]

pub const INT_MIN:       i32 = int::min;
pub const INT_MAX:       i32 = int::max;
pub const LONG_MIN:      i64 = i64::min;
pub const LONG_MAX:      i64 = i64::max;
pub const FLOAT_LOWEST:  f32 = f32::lowest;
pub const FLOAT_MIN:     f32 = f32::min;
pub const FLOAT_MAX:     f32 = f32::max;
pub const DOUBLE_LOWEST: f64 = f64::lowest;
pub const DOUBLE_MIN:    f64 = f64::min;
pub const DOUBLE_MAX:    f64 = f64::max;

lazy_static!{
    /*
    const vector<int> ints {
      int_min,
      int_min + 1,
      int_min + 2,
      static_cast<int>(-sqrt(int_max)),
      -3, -2, -1, 0, 1, 2, 3,
      static_cast<int>(sqrt(int_max)),
      int_max - 2,
      int_max - 1,
      int_max
    };
    const vector<int> non_neg_ints {
      0, 1, 2, 3,
      static_cast<int>(sqrt(int_max)),
      int_max - 2,
      int_max - 1,
      int_max
    };
    const vector<i64> longs {
      long_min,
      long_min + 1,
      long_min + 2,
      static_cast<i64>(-sqrt(long_max)),
      -3, -2, -1, 0, 1, 2, 3,
      static_cast<i64>(sqrt(long_max)),
      long_max - 2,
      long_max - 1,
      long_max
    };
    const vector<i64> non_neg_longs {
      0, 1, 2, 3,
      static_cast<i64>(sqrt(long_max)),
      long_max - 2,
      long_max - 1,
      long_max
    };
    const vector<float> floats {
      float_lowest,
      -3.0f, -2.0f, -1.0f, -1.0f/2.0f, -1.0f/3.0f,
      -float_min,
      0.0,
      float_min,
      1.0f/3.0f, 1.0f/2.0f, 1.0f, 2.0f, 3.0f,
      float_max,
    };
    const vector<double> doubles {
      double_lowest,
      -3.0, -2.0, -1.0, -1.0/2.0, -1.0/3.0,
      -double_min,
      0.0,
      double_min,
      1.0/3.0, 1.0/2.0, 1.0, 2.0, 3.0,
      double_max,
    };
    */
}

pub fn assert_eq_float<T: Float>(
        val: T,
        act: T,
        exp: T) {
    
    todo!();
        /*
            if (isnan(act) || isnan(exp)) {
        return;
      }
      ASSERT_FLOAT_EQ(act, exp);
        */
}

pub fn assert_eq_int<T: Integer>(
        val: T,
        act: T,
        exp: T)  {
    
    todo!();
        /*
            if (val != 0 && act == 0) {
        return;
      }
      if (val != 0 && exp == 0) {
        return;
      }
      const auto min = T::min;
      if (exp == min && val != min) {
        return;
      }
      ASSERT_EQ(act, exp);
        */
}

pub fn typed_pow_float<T: Float>(base: T, exp: T) -> T {
    
    todo!();
        /*
            return pow(base, exp);
        */
}

pub fn typed_pow_int<T: Integer>(base: T, exp: T) -> T {
    
    todo!();
        /*
            return native::powi(base, exp);
        */
}

pub fn tensor_pow_scalar<Vals, Pows>(
    vals:       Vals,
    pows:       Pows,
    vals_dtype: TorchScalarType,
    dtype:      TorchScalarType)  {

    todo!();
        /*
            const auto tensor = Torchtensor(vals, valsDtype);

      for (const auto pow : pows) {
        if ( dtype == kInt && pow > int::max) {
          // value cannot be converted to type int without overflow
          EXPECT_THROW(tensor.pow(pow), runtime_error);
          continue;
        }
        auto actual_pow = tensor.pow(pow);

        auto actual_pow_ = Torchempty_like(actual_pow);
        actual_pow_.copy_(tensor);
        actual_pow_.pow_(pow);

        auto actual_pow_out = Torchempty_like(actual_pow);
        Torchpow_out(actual_pow_out, tensor, pow);

        auto actual_torch_pow = Torchpow(tensor, pow);

        int i = 0;
        for (const auto val : vals) {
          const auto exp = Torchpow(Torchtensor({val}, dtype), Torchtensor(pow, dtype)).template item<double>();

          const auto act_pow = actual_pow[i].to(kDouble).template item<double>();
          assert_eq<long double>(val, act_pow, exp);

          const auto act_pow_ = actual_pow_[i].to(kDouble).template item<double>();
          assert_eq<long double>(val, act_pow_, exp);

          const auto act_pow_out = actual_pow_out[i].to(kDouble).template item<double>();
          assert_eq<long double>(val, act_pow_out, exp);

          const auto act_torch_pow = actual_torch_pow[i].to(kDouble).template item<double>();
          assert_eq<long double>(val, act_torch_pow, exp);

          i++;
        }
      }
        */
}

pub fn scalar_pow_tensor<Vals, Pows>(
    vals:       Vals,
    vals_dtype: ScalarType,
    pows:       Pows,
    pows_dtype: ScalarType)  {

    todo!();
        /*
            using T = typename Pows::value_type;

      const auto pow_tensor = Torchtensor(pows, pows_dtype);

      for (const auto val : vals) {
        const auto actual_pow = Torchpow(val, pow_tensor);
        auto actual_pow_out1 = Torchempty_like(actual_pow);
        const auto actual_pow_out2 =
          Torchpow_out(actual_pow_out1, val, pow_tensor);

        int i = 0;
        for (const auto pow : pows) {
          const auto exp = typed_pow(static_cast<T>(val), T(pow));

          const auto act_pow = actual_pow[i].template item<T>();
          assert_eq<T>(val, act_pow, exp);

          const auto act_pow_out1 = actual_pow_out1[i].template item<T>();
          assert_eq<T>(val, act_pow_out1, exp);

          const auto act_pow_out2 = actual_pow_out2[i].template item<T>();
          assert_eq<T>(val, act_pow_out2, exp);

          i++;
        }
      }
        */
}

pub fn tensor_pow_tensor<Vals, Pows>(
    vals:       Vals,
    vals_dtype: ScalarType,
    pows:       Pows,
    pows_dtype: ScalarType)  {

    todo!();
        /*
            using T = typename Vals::value_type;

      typedef numeric_limits< double > dbl;
      cout.precision(dbl::max_digits10);

      const auto vals_tensor = Torchtensor(vals, vals_dtype);
      for (usize shift = 0; shift < pows.size(); shift++) {
        const auto pows_tensor = Torchtensor(pows, pows_dtype);

        const auto actual_pow = vals_tensor.pow(pows_tensor);

        auto actual_pow_ = vals_tensor.clone();
        actual_pow_.pow_(pows_tensor);

        auto actual_pow_out = Torchempty_like(vals_tensor);
        Torchpow_out(actual_pow_out, vals_tensor, pows_tensor);

        auto actual_torch_pow = Torchpow(vals_tensor, pows_tensor);

        int i = 0;
        for (const auto val : vals) {
          const auto pow = pows[i];
          const auto exp = typed_pow(T(val), T(pow));

          const auto act_pow = actual_pow[i].template item<T>();
          assert_eq(val, act_pow, exp);

          const auto act_pow_ = actual_pow_[i].template item<T>();
          assert_eq(val, act_pow_, exp);

          const auto act_pow_out = actual_pow_out[i].template item<T>();
          assert_eq(val, act_pow_out, exp);

          const auto act_torch_pow = actual_torch_pow[i].template item<T>();
          assert_eq(val, act_torch_pow, exp);

          i++;
        }

        rotate(pows.begin(), pows.begin() + 1, pows.end());
      }
        */
}

pub fn test_pow_one<T>(vals: Vec<T>)  {

    todo!();
        /*
            for (const auto val : vals) {
        ASSERT_EQ(native::powi(val, T(1)), val);
      }
        */
}

pub fn test_squared<T>(vals: Vec<T>)  {

    todo!();
        /*
            for (const auto val : vals) {
        ASSERT_EQ(native::powi(val, T(2)), val * val);
      }
        */
}

pub fn test_cubed<T>(vals: Vec<T>)  {

    todo!();
        /*
            for (const auto val : vals) {
        ASSERT_EQ(native::powi(val, T(3)), val * val * val);
      }
        */
}

pub fn test_inverse<T>(vals: Vec<T>)  {

    todo!();
        /*
            for (const auto val : vals) {
        // 1 has special checks below
        if ( val != 1 && val != -1) {
          ASSERT_EQ(native::powi(val, T(-4)), 0);
          ASSERT_EQ(native::powi(val, T(-1)), val==1);
        }
      }
      T neg1 = -1;
      ASSERT_EQ(native::powi(neg1, T(0)), 1);
      ASSERT_EQ(native::powi(neg1, T(-1)), -1);
      ASSERT_EQ(native::powi(neg1, T(-2)), 1);
      ASSERT_EQ(native::powi(neg1, T(-3)), -1);
      ASSERT_EQ(native::powi(neg1, T(-4)), 1);

      T one = 1;
      ASSERT_EQ(native::powi(one, T(0)), 1);
      ASSERT_EQ(native::powi(one, T(-1)), 1);
      ASSERT_EQ(native::powi(one, T(-2)), 1);
      ASSERT_EQ(native::powi(one, T(-3)), 1);
      ASSERT_EQ(native::powi(one, T(-4)), 1);
        */
}

#[test] fn pow_test_int_tensor_all_scalars() {
    todo!();
    /*
    
      tensor_pow_scalar(ints, non_neg_ints, kInt, kInt);
      tensor_pow_scalar(ints, non_neg_longs, kInt, kInt);
      tensor_pow_scalar(ints, floats, kInt, kFloat);
      tensor_pow_scalar(ints, doubles, kInt, kDouble);

    */
}

#[test] fn pow_test_long_tensor_all_scalars() {
    todo!();
    /*
    
      tensor_pow_scalar(longs, non_neg_ints, kLong, kLong);
      tensor_pow_scalar(longs, non_neg_longs, kLong, kLong);
      tensor_pow_scalar(longs, floats, kLong, kFloat);
      tensor_pow_scalar(longs, doubles, kLong, kDouble);

    */
}

#[test] fn pow_test_float_tensor_all_scalars() {
    todo!();
    /*
    
      tensor_pow_scalar(floats, ints, kFloat, kDouble);
      tensor_pow_scalar(floats, longs, kFloat, kDouble);
      tensor_pow_scalar(floats, floats, kFloat, kFloat);
      tensor_pow_scalar(floats, doubles, kFloat, kDouble);

    */
}

#[test] fn pow_test_double_tensor_all_scalars() {
    todo!();
    /*
    
      tensor_pow_scalar(doubles, ints, kDouble, kDouble);
      tensor_pow_scalar(doubles, longs, kDouble, kDouble);
      tensor_pow_scalar(doubles, floats, kDouble, kDouble);
      tensor_pow_scalar(doubles, doubles, kDouble, kDouble);

    */
}

#[test] fn pow_test_int_scalar_all_tensors() {
    todo!();
    /*
    
      scalar_pow_tensor(ints, kInt, ints, kInt);
      scalar_pow_tensor(ints, kInt, longs, kLong);
      scalar_pow_tensor(ints, kInt, floats, kFloat);
      scalar_pow_tensor(ints, kInt, doubles, kDouble);

    */
}

#[test] fn pow_test_long_scalar_all_tensors() {
    todo!();
    /*
    
      scalar_pow_tensor(longs, kLong, longs, kLong);
      scalar_pow_tensor(longs, kLong, floats, kFloat);
      scalar_pow_tensor(longs, kLong, doubles, kDouble);

    */
}

#[test] fn pow_test_float_scalar_all_tensors() {
    todo!();
    /*
    
      scalar_pow_tensor(floats, kFloat, floats, kFloat);
      scalar_pow_tensor(floats, kFloat, doubles, kDouble);

    */
}

#[test] fn pow_test_double_scalar_all_tensors() {
    todo!();
    /*
    
      scalar_pow_tensor(doubles, kDouble, doubles, kDouble);

    */
}

#[test] fn pow_test_int_tensor() {
    todo!();
    /*
    
      tensor_pow_tensor(ints, kInt, ints, kInt);

    */
}

#[test] fn pow_test_long_tensor() {
    todo!();
    /*
    
      tensor_pow_tensor(longs, kLong, longs, kLong);

    */
}

#[test] fn pow_test_float_tensor() {
    todo!();
    /*
    
      tensor_pow_tensor(floats, kFloat, floats, kFloat);

    */
}

#[test] fn pow_test_double_tensor() {
    todo!();
    /*
    
      tensor_pow_tensor(doubles, kDouble, doubles, kDouble);

    */
}

#[test] fn pow_test_integral() {
    todo!();
    /*
    
      test_pow_one(longs);
      test_pow_one(ints);

      test_squared(longs);
      test_squared(ints);

      test_cubed(longs);
      test_cubed(ints);

      test_inverse(longs);
      test_inverse(ints);

    */
}
