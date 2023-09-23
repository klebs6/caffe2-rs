// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/lut-microkernel-tester.h]

pub struct LUTMicrokernelTester {
    n:          usize, // default = { 1 }
    inplace:    bool,  // default = { false }
    iterations: usize, // default = { 15 }
}

impl LUTMicrokernelTester {

    
    #[inline] pub fn n(&mut self, n: usize) -> &mut LUTMicrokernelTester {
        
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
    
    #[inline] pub fn inplace(&mut self, inplace: bool) -> &mut LUTMicrokernelTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut LUTMicrokernelTester {
        
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
    
    pub fn test(&self, x8lut: PyTorchX8LutUKernelFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> x(n());
        vector<u8> t(256);
        vector<u8> y(n());
        vector<u8> yRef(n());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(x.begin(), x.end(), ref(u8rng));
          generate(t.begin(), t.end(), ref(u8rng));
          if (inplace()) {
            generate(y.begin(), y.end(), ref(u8rng));
          } else {
            fill(y.begin(), y.end(), 0xA5);
          }
          const u8* xData = inplace() ? y.data() : x.data();

          /* Compute reference results */
          for (usize i = 0; i < n(); i++) {
            yRef[i] = t[xData[i]];
          }

          /* Call optimized micro-kernel */
          x8lut(n(), xData, t.data(), y.data());

          /* Verify results */
          for (usize i = 0; i < n(); i++) {
            ASSERT_EQ(u32(yRef[i]), u32(y[i]))
                << "at position " << i << ", n = " << n();
          }
        }
        */
    }
}
