crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/util/bfloat16_test.cpp]

pub fn float_from_bytes(
        sign:     u32,
        exponent: u32,
        fraction: u32) -> f32 {
    
    todo!();
        /*
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      uint32_t bytes;
      bytes = 0;
      bytes |= sign;
      bytes <<= 8;
      bytes |= exponent;
      bytes <<= 23;
      bytes |= fraction;

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      float res;
      memcpy(&res, &bytes, sizeof(res));
      return res;
        */
}

#[test] fn b_float_16conversion_to_bfloat_16and_back() {
    todo!();
    /*
    
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
      float in[100];
      for (int i = 0; i < 100; ++i) {
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers)
        in[i] = i + 1.25;
      }

      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
      BFloat16 bfloats[100];
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
      float out[100];

      for (int i = 0; i < 100; ++i) {
        bfloats[i].x = bits_from_f32(in[i]);
        out[i] = f32_from_bits(bfloats[i].x);

        // The relative error should be less than 1/(2^7) since BFloat16
        // has 7 bits mantissa.
        EXPECT_LE(fabs(out[i] - in[i]) / in[i], 1.0 / 128);
      }

    */
}

#[test] fn b_float_16conversion_to_bfloat_16rne_and_back() {
    todo!();
    /*
    
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
      float in[100];
      for (int i = 0; i < 100; ++i) {
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers)
        in[i] = i + 1.25;
      }

      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
      BFloat16 bfloats[100];
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
      float out[100];

      for (int i = 0; i < 100; ++i) {
        bfloats[i].x = round_to_nearest_even(in[i]);
        out[i] = f32_from_bits(bfloats[i].x);

        // The relative error should be less than 1/(2^7) since BFloat16
        // has 7 bits mantissa.
        EXPECT_LE(fabs(out[i] - in[i]) / in[i], 1.0 / 128);
      }

    */
}

#[test] fn b_float_16conversion_nan() {
    todo!();
    /*
    
      float inNaN = float_from_bytes(0, 0xFF, 0x7FFFFF);
      EXPECT_TRUE(isnan(inNaN));

      BFloat16 a = BFloat16(inNaN);
      float out = f32_from_bits(a.x);

      EXPECT_TRUE(isnan(out));

    */
}

#[test] fn b_float_16conversion_inf() {
    todo!();
    /*
    
      float inInf = float_from_bytes(0, 0xFF, 0);
      EXPECT_TRUE(isinf(inInf));

      BFloat16 a = BFloat16(inInf);
      float out = f32_from_bits(a.x);

      EXPECT_TRUE(isinf(out));

    */
}

#[test] fn b_float_16conversion_smallest_denormal() {
    todo!();
    /*
    
      float in = numeric_limits<float>::denorm_min(); // The smallest non-zero
                                                           // subnormal number
      BFloat16 a = BFloat16(in);
      float out = f32_from_bits(a.x);

      EXPECT_FLOAT_EQ(in, out);

    */
}

#[test] fn b_float_16math_addition() {
    todo!();
    /*
    
      // This test verifies that if only first 7 bits of float's mantissa are
      // changed after addition, we should have no loss in precision.

      // input bits
      // S | Exponent | Mantissa
      // 0 | 10000000 | 10010000000000000000000 = 3.125
      float input = float_from_bytes(0, 0, 0x40480000);

      // expected bits
      // S | Exponent | Mantissa
      // 0 | 10000001 | 10010000000000000000000 = 6.25
      float expected = float_from_bytes(0, 0, 0x40c80000);

      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      BFloat16 b;
      b.x = bits_from_f32(input);
      b = b + b;

      float res = f32_from_bits(b.x);
      EXPECT_EQ(res, expected);

    */
}

#[test] fn b_float_16math_subtraction() {
    todo!();
    /*
    
      // This test verifies that if only first 7 bits of float's mantissa are
      // changed after subtraction, we should have no loss in precision.

      // input bits
      // S | Exponent | Mantissa
      // 0 | 10000001 | 11101000000000000000000 = 7.625
      float input = float_from_bytes(0, 0, 0x40f40000);

      // expected bits
      // S | Exponent | Mantissa
      // 0 | 10000000 | 01010000000000000000000 = 2.625
      float expected = float_from_bytes(0, 0, 0x40280000);

      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      BFloat16 b;
      b.x = bits_from_f32(input);
      b = b - 5;

      float res = f32_from_bits(b.x);
      EXPECT_EQ(res, expected);

    */
}

pub fn binary_to_float(bytes: u32) -> f32 {
    
    todo!();
        /*
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      float res;
      memcpy(&res, &bytes, sizeof(res));
      return res;
        */
}

#[cfg(test)]
mod bfloat16_test {

    pub struct BFloat16TestParam {
        input: u32,
        rne:   u16,
    }

    fn b_float_16test_16rne_test(input: BFloat16TestParam) {

        todo!();

        /*
          float value = BinaryToFloat(GetParam().input);
          u16 rounded = round_to_nearest_even(value);
          EXPECT_EQ(GetParam().rne, rounded);

        */
    }

    #[test] fn b_float_16test_16rne_test_all() {

        b_float_16test_16rne_test(BFloat16TestParam { input: 0x3F848000, rne: 0x3F84});
        b_float_16test_16rne_test(BFloat16TestParam { input: 0x3F848010, rne: 0x3F85});
        b_float_16test_16rne_test(BFloat16TestParam { input: 0x3F850000, rne: 0x3F85});
        b_float_16test_16rne_test(BFloat16TestParam { input: 0x3F858000, rne: 0x3F86});
        b_float_16test_16rne_test(BFloat16TestParam { input: 0x3FFF8000, rne: 0x4000});
    }
}
