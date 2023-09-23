crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/rmax-microkernel-tester.h]

pub struct RMaxMicrokernelTester {
    n:          usize, // default = { 1 }
    iterations: usize, // default = { 15 }
}

impl RMaxMicrokernelTester {
    
    #[inline] pub fn n(&mut self, n: usize) -> &mut RMaxMicrokernelTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut RMaxMicrokernelTester {
        
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
    
    pub fn test(&self, u8rmax: PyTorchU8RMaxUKernelFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> x(n());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(x.begin(), x.end(), ref(u8rng));

          /* Compute reference results */
          u8 yRef = 0;
          for (usize i = 0; i < n(); i++) {
            yRef = max(yRef, x[i]);
          }

          /* Call optimized micro-kernel */
          const u8 y = u8rmax(n(), x.data());

          /* Verify results */
          ASSERT_EQ(yRef, y) << "n = " << n();
        }
        */
    }
}
