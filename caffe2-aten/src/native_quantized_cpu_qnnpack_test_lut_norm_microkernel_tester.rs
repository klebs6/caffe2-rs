// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/lut-norm-microkernel-tester.h]

pub struct LUTNormMicrokernelTester {
    n:          usize, // default = { 1 }
    inplace:    bool, // default = { false }
    iterations: usize, // default = { 15 }
}

impl LUTNormMicrokernelTester {

    #[inline] pub fn n(&mut self, n: usize) -> &mut LUTNormMicrokernelTester {
        
        todo!();
        /*
            assert(n != 0);
        this->n_ = n;
        return *this;
        */
    }
    
    #[inline] pub fn n(&self) -> usize {
        
        todo!();
        /*
            return this->n_;
        */
    }
    
    #[inline] pub fn inplace(&mut self, inplace: bool) -> &mut LUTNormMicrokernelTester {
        
        todo!();
        /*
            this->inplace_ = inplace;
        return *this;
        */
    }
    
    #[inline] pub fn inplace(&self) -> bool {
        
        todo!();
        /*
            return this->inplace_;
        */
    }
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut LUTNormMicrokernelTester {
        
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
    
    pub fn test(&self, u8lut_32norm: PyTorchU8Lut32NormUKernelFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);
        auto u32rng = bind(
            uniform_int_distribution<u32>(
                1, u32::max / (257 * n())),
            rng);

        vector<u8> x(n());
        vector<u32> t(256);
        vector<u8> y(n());
        vector<float> yRef(n());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(x.begin(), x.end(), ref(u8rng));
          generate(t.begin(), t.end(), ref(u32rng));
          if (inplace()) {
            generate(y.begin(), y.end(), ref(u8rng));
          } else {
            fill(y.begin(), y.end(), 0xA5);
          }
          const u8* xData = inplace() ? y.data() : x.data();

          /* Compute reference results */
          u32 sum = 0;
          for (usize i = 0; i < n(); i++) {
            sum += t[xData[i]];
          }
          for (usize i = 0; i < n(); i++) {
            yRef[i] = 256.0f * float(t[xData[i]]) / float(sum);
            yRef[i] = min(yRef[i], 255.0f);
          }

          /* Call optimized micro-kernel */
          u8lut32norm(n(), xData, t.data(), y.data());

          /* Verify results */
          for (usize i = 0; i < n(); i++) {
            ASSERT_NEAR(yRef[i], float(y[i]), 0.5f)
                << "at position " << i << ", n = " << n() << ", sum = " << sum;
          }
        }
        */
    }
}
