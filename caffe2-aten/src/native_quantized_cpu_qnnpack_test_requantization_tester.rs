crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/requantization-tester.h]

pub struct RequantizationTester {

    zero_point: usize, // default = { 0 }
    s:          usize, // default = { 1 }

    /**
      | {u8::min};
      |
      */
    qmin:       u8,


    /**
      | {u8::max};
      |
      */
    qmax:       u8,

    iterations: usize, // default = { 1 }
}

impl RequantizationTester {
    
    #[inline] pub fn s(&mut self, s: u32) -> &mut RequantizationTester {
        
        todo!();
        /*
            this->s_ = s;
        return *this;
        */
    }
    
    #[inline] pub fn s(&self) -> u32 {
        
        todo!();
        /*
            return this->s_;
        */
    }
    
    #[inline] pub fn scale(&self) -> f32 {
        
        todo!();
        /*
            return ldexpf(1.0f, -s());
        */
    }
    
    #[inline] pub fn zero_point(&mut self, zero_point: i32) -> &mut RequantizationTester {
        
        todo!();
        /*
            this->zeroPoint_ = zeroPoint;
        return *this;
        */
    }
    
    #[inline] pub fn zero_point(&self) -> i32 {
        
        todo!();
        /*
            return this->zeroPoint_;
        */
    }
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut RequantizationTester {
        
        todo!();
        /*
            this->qmin_ = qmin;
        return *this;
        */
    }
    
    #[inline] pub fn qmin(&self) -> u8 {
        
        todo!();
        /*
            return this->qmin_;
        */
    }
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut RequantizationTester {
        
        todo!();
        /*
            this->qmax_ = qmax;
        return *this;
        */
    }
    
    #[inline] pub fn qmax(&self) -> u8 {
        
        todo!();
        /*
            return this->qmax_;
        */
    }
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut RequantizationTester {
        
        todo!();
        /*
            this->iterations_ = iterations;
        return *this;
        */
    }
    
    #[inline] pub fn iterations(&self) -> usize {
        
        todo!();
        /*
            return this->iterations_;
        */
      }

    /**
      | Test that requantization of numbers
      | ((i - zero point) * 2**s) with
      | 
      | - scale = exp2(-s)
      | 
      | - zero point in [0, 255]
      | 
      | - no output clamping produces exactly
      | i, provided that ((i - zero point) * 2**s)
      | does not overflow.
      |
      */
    pub fn test_exact_divide_bypo2(&self, requantize: PyTorchRequantizationFunction)  {
        
        todo!();
        /*
            ASSERT_GE(zeroPoint(), 0);
            ASSERT_LE(zeroPoint(), 255);

            /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
            ASSERT_GE(s(), 1);
            ASSERT_LT(s(), 32);

            vector<i32> inputs(256);
            vector<u8> outputs(inputs.size());
            const i32 maxI =
                (u32(i32::max) >> s()) + zeroPoint();
            const i32 minI =
                -(-u32(i32::min) >> s()) + zeroPoint();
            for (i32 i = 0; i < 256; i++) {
              const i32 clampedI = max(minI, min(maxI, i));
              inputs[i] = i32(u32(clampedI - zeroPoint()) << s());
            }
            requantize(
                inputs.size(),
                inputs.data(),
                scale(),
                zeroPoint(),
                qmin(),
                qmax(),
                outputs.data());
            for (i32 i = 0; i < 256; i++) {
              const i32 clampedI = max(minI, min(maxI, i));
              ASSERT_EQ(clampedI, outputs[i])
                  << "i = " << i << ", clamped i = " << clampedI << ", min i = " << minI
                  << ", max i = " << maxI << ", s = " << s()
                  << ", zero point = " << zeroPoint();
            }
        */
    }

    /**
      | Test that requantization of numbers
      | (i * 2**s + sign(i - zero point) * 2**(s-1))
      | with
      | 
      | - scale = exp2(-s)
      | 
      | - zero point in [1, 255]
      | 
      | - no output clamping produces exactly
      | i, provided that ((i - zero point) * 2**s)
      | does not overflow.
      |
      */
    pub fn test_divide_by_po2with_rounding_up(&mut self, requantize: PyTorchRequantizationFunction)  {
        
        todo!();
        /*
            ASSERT_GE(zeroPoint(), 0);
            ASSERT_LE(zeroPoint(), 255);

            /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
            ASSERT_GE(s(), 1);
            ASSERT_LT(s(), 32);

            vector<i32> inputs(256);
            vector<u8> outputs(inputs.size());
            for (i32 i = 0; i < 256; i++) {
              const i64 input =
                  RequantizationTester::shiftLeft(i - zeroPoint(), s()) -
                  (INT64_C(1) << (s() - 1)) + (i64)(i <= zeroPoint());
              inputs[i] = i32(input);
            }
            requantize(
                inputs.size(),
                inputs.data(),
                scale(),
                zeroPoint(),
                qmin(),
                qmax(),
                outputs.data());
            for (i32 i = 0; i < 256; i++) {
              const i64 input =
                  RequantizationTester::shiftLeft(i - zeroPoint(), s()) -
                  (INT64_C(1) << (s() - 1)) + (i64)(i <= zeroPoint());
              if (i32(input) == input) {
                ASSERT_EQ(i, u32(outputs[i]))
                    << "i = " << i << ", input = " << input << ", s = " << s()
                    << ", zero point = " << zeroPoint();
              }
            }
        */
    }

    /**
      | Test that requantization of numbers
      | (i * 2**s + sign(i - zero point) * 2**(s-1))
      | with
      | 
      | - scale = exp2(-s)
      | 
      | - zero point in [1, 255]
      | 
      | - no output clamping produces exactly
      | i, provided that ((i - zero point) * 2**s)
      | does not overflow.
      |
      */
    pub fn test_divide_by_po2with_rounding_down(&mut self, requantize: PyTorchRequantizationFunction)  {
        
        todo!();
        /*
            ASSERT_GE(zeroPoint(), 0);
            ASSERT_LE(zeroPoint(), 255);

            /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
            ASSERT_GE(s(), 1);
            ASSERT_LT(s(), 32);

            vector<i32> inputs(256);
            vector<u8> outputs(inputs.size());
            for (i32 i = 0; i < 256; i++) {
              const i64 input =
                  RequantizationTester::shiftLeft(i - zeroPoint(), s()) +
                  (INT64_C(1) << (s() - 1)) - (i64)(i >= zeroPoint());
              inputs[i] = i32(input);
            }
            requantize(
                inputs.size(),
                inputs.data(),
                scale(),
                zeroPoint(),
                qmin(),
                qmax(),
                outputs.data());
            for (i32 i = 0; i < 256; i++) {
              const i64 input =
                  RequantizationTester::shiftLeft(i - zeroPoint(), s()) +
                  (INT64_C(1) << (s() - 1)) - (i64)(i >= zeroPoint());
              if (i32(input) == input) {
                ASSERT_EQ(i, u32(outputs[i]))
                    << "i = " << i << ", input = " << input << ", s = " << s()
                    << ", zero point = " << zeroPoint();
              }
            }
        */
    }
    
    pub fn test_divide_by_po2with_rounding_away(&mut self, requantize: PyTorchRequantizationFunction)  {
        
        todo!();
        /*
            ASSERT_GE(zeroPoint(), 0);
            ASSERT_LE(zeroPoint(), 255);

            /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
            ASSERT_GE(s(), 1);
            ASSERT_LT(s(), 32);

            vector<i32> inputs(256);
            vector<u8> outputs(inputs.size());
            for (i32 i = 0; i < 256; i++) {
              i64 input = RequantizationTester::shiftLeft(i - zeroPoint(), s());
              if (input > 0) {
                input -= INT64_C(1) << (s() - 1);
              } else if (input < 0) {
                input += INT64_C(1) << (s() - 1);
              }
              inputs[i] = i32(input);
            }
            requantize(
                inputs.size(),
                inputs.data(),
                scale(),
                zeroPoint(),
                qmin(),
                qmax(),
                outputs.data());
            for (u32 i = 0; i < 256; i++) {
              i64 input = RequantizationTester::shiftLeft(i - zeroPoint(), s());
              if (input > 0) {
                input -= INT64_C(1) << (s() - 1);
              } else if (input < 0) {
                input += INT64_C(1) << (s() - 1);
              }
              if (i32(input) == input) {
                ASSERT_EQ(i, u32(outputs[i]))
                    << "i = " << i << ", input = " << input << ", s = " << s()
                    << ", zero point = " << zeroPoint();
              }
            }
        */
    }
    
    pub fn test_special_cases(&mut self, requantize: PyTorchRequantizationFunction)  {
        
        todo!();
        /*
            vector<i32> inputs(256);
            vector<u8> outputs(inputs.size());

            fill(
                inputs.begin(), inputs.end(), i32::min);
            for (i32 zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
              requantize(
                  inputs.size(),
                  inputs.data(),
                  ldexpf(1.0f, -32) /* scale */,
                  zeroPoint /* zero point */,
                  u8::min,
                  u8::max,
                  outputs.data());
              ASSERT_EQ(
                  max(i32(0), zeroPoint - 1),
                  *min_element(outputs.cbegin(), outputs.cend()));
            }

            fill(
                inputs.begin(), inputs.end(), i32::max);
            requantize(
                inputs.size(),
                inputs.data(),
                0x1.FFFFFEp-1f /* scale */,
                u8::max /* zero point */,
                u8::min,
                u8::max,
                outputs.data());
            for (usize i = 0; i < inputs.size(); i++) {
              ASSERT_EQ(u8::max, outputs[i]);
            }
        */
    }
    
    pub fn test_random_cases_precise(&mut self, requantize: PyTorchRequantizationFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
            mt19937 mtRng(randomDevice());
            for (usize iteration = 0; iteration < iterations(); iteration++) {
              auto rng = bind(uniform_int_distribution<u8>(), mtRng);

              vector<i32> inputs(4096);
              vector<u8> outputs(inputs.size());

              const u8 zeroPoint = UINT8_C(128);
              uniform_real_distribution<float> scaleDistribution(
                  0x1.000000p-23f, 0x1.FFFFFEp-1f);
              const float scale = scaleDistribution(mtRng);
              for (usize i = 0; i < inputs.size(); i++) {
                const u8 approximateOutput = rng();
                const i32 input =
                    i32(double(approximateOutput) / double(scale));
                inputs[i] = input;
              }

              requantize(
                  inputs.size(),
                  inputs.data(),
                  scale,
                  zeroPoint,
                  u8::min,
                  u8::max,
                  outputs.data());

              /* Ensure that outputs are not all identical, as in this case test doesn't
               * validate much */
              ASSERT_NE(
                  *max_element(outputs.cbegin(), outputs.cend()),
                  *min_element(outputs.cbegin(), outputs.cend()));

              for (usize i = 0; i < inputs.size(); i++) {
                const u8 referenceOutput = pytorch_scalar_requantize_precise(
                    inputs[i],
                    scale,
                    zeroPoint,
                    u8::min,
                    u8::max);
                ASSERT_EQ(u32(referenceOutput), u32(outputs[i]));
              }
            }
        */
    }
    
    pub fn test_random_cases_approximate(&mut self, requantize: PyTorchRequantizationFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
            mt19937 mtRng(randomDevice());
            for (usize iteration = 0; iteration < iterations(); iteration++) {
              auto rng = bind(uniform_int_distribution<u8>(), mtRng);

              vector<i32> inputs(4096);
              vector<u8> outputs(inputs.size());

              const u8 zeroPoint = UINT8_C(128);
              uniform_real_distribution<float> scaleDistribution(
                  0x1.000000p-23f, 0x1.FFFFFEp-1f);
              const float scale = scaleDistribution(mtRng);
              for (usize i = 0; i < inputs.size(); i++) {
                const u8 approximateOutput = rng();
                const i32 input =
                    i32(double(approximateOutput) / double(scale));
                inputs[i] = input;
              }

              requantize(
                  inputs.size(),
                  inputs.data(),
                  scale,
                  zeroPoint,
                  u8::min,
                  u8::max,
                  outputs.data());

              /* Ensure that outputs are not all identical, as in this case test doesn't
               * validate much */
              ASSERT_NE(
                  *max_element(outputs.cbegin(), outputs.cend()),
                  *min_element(outputs.cbegin(), outputs.cend()));

              for (usize i = 0; i < inputs.size(); i++) {
                const double referenceOutput =
                    RequantizationTester::requantizeApproximate(
                        inputs[i],
                        scale,
                        zeroPoint,
                        u8::min,
                        u8::max);
                ASSERT_LE(fabs(referenceOutput - double(outputs[i])), 0.55)
                    << "input = " << inputs[i] << ", output = " << u32(outputs[i])
                    << ", reference output = " << referenceOutput;
              }
            }
        */
    }
    
    pub fn test_random_cases_against_reference(&mut self, 
        requantize:           PyTorchRequantizationFunction,
        requantize_reference: PyTorchRequantizationFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
            mt19937 mtRng(randomDevice());
            for (usize iteration = 0; iteration < iterations(); iteration++) {
              auto rng = bind(uniform_int_distribution<u8>(), mtRng);

              vector<i32> inputs(4096);
              vector<u8> outputs(inputs.size());
              vector<u8> referenceOutputs(inputs.size());

              const u8 zeroPoint = UINT8_C(128);
              uniform_real_distribution<float> scaleDistribution(
                  0x1.000000p-23f, 0x1.FFFFFEp-1f);
              const float scale = scaleDistribution(mtRng);
              for (usize i = 0; i < inputs.size(); i++) {
                const u8 approximateOutput = rng();
                const i32 input =
                    i32(double(approximateOutput) / double(scale));
                inputs[i] = input;
              }

              requantize(
                  inputs.size(),
                  inputs.data(),
                  scale,
                  zeroPoint,
                  u8::min,
                  u8::max,
                  outputs.data());

              requantizeReference(
                  inputs.size(),
                  inputs.data(),
                  scale,
                  zeroPoint,
                  u8::min,
                  u8::max,
                  referenceOutputs.data());

              /* Ensure that outputs are not all identical, as in this case test doesn't
               * validate much */
              ASSERT_NE(
                  *max_element(outputs.cbegin(), outputs.cend()),
                  *min_element(outputs.cbegin(), outputs.cend()));

              for (usize i = 0; i < inputs.size(); i++) {
                ASSERT_EQ(u32(referenceOutputs[i]), u32(outputs[i]));
              }
            }
        */
    }
    
    #[inline] pub fn shift_left(w: i64, n: u32) -> i64 {
        
        todo!();
        /*
            return (i64)((u64)w << n);
        */
    }
    
    #[inline] pub fn requantize_approximate(
        value:      i32,
        scale:      f32,
        zero_point: u8,
        qmin:       u8,
        qmax:       u8) -> f64 {
        
        todo!();
        /*
            assert(scale < 1.0f);
            assert(scale >= 0x1.0p-32f);

            double clampedValue = double(value) * double(scale) + double(zeroPoint);

            const double fmin = double(qmin);
            if (clampedValue < fmin) {
              clampedValue = fmin;
            }

            const double fmax = double(qmax);
            if (clampedValue > fmax) {
              clampedValue = fmax;
            }

            return clampedValue;
        */
    }
}
