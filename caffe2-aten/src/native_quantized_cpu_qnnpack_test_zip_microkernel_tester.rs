crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/zip-microkernel-tester.h]

pub struct ZipMicrokernelTester {
    n:          usize, // default = { 1 }
    g:          usize, // default = { 1 }
    iterations: usize, // default = { 3 }
}

impl ZipMicrokernelTester {
    
    #[inline] pub fn n(&mut self, n: usize) -> &mut ZipMicrokernelTester {
        
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
    
    #[inline] pub fn g(&mut self, g: usize) -> &mut ZipMicrokernelTester {
        
        todo!();
        /*
            assert(g != 0);
        this->g_ = g;
        return *this;
        */
    }
    
    #[inline] pub fn g(&self) -> usize {
        
        todo!();
        /*
            return this->g_;
        */
    }
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut ZipMicrokernelTester {
        
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
    
    pub fn test(&self, xzip: PyTorchXZipcUKernelFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> x(n() * g());
        vector<u8> y(g() * n());

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(x.begin(), x.end(), ref(u8rng));
          fill(y.begin(), y.end(), 0xA5);

          /* Call optimized micro-kernel */
          xzip(n(), x.data(), y.data());

          /* Verify results */
          for (usize i = 0; i < n(); i++) {
            for (usize j = 0; j < g(); j++) {
              ASSERT_EQ(u32(y[i * g() + j]), u32(x[j * n() + i]))
                  << "at element " << i << ", group " << j;
            }
          }
        }
        */
    }
    
    pub fn test(&self, xzip: PyTorchXZipvUKernelFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> x(n() * g());
        vector<u8> y(g() * n());

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(x.begin(), x.end(), ref(u8rng));
          fill(y.begin(), y.end(), 0xA5);

          /* Call optimized micro-kernel */
          xzip(n(), g(), x.data(), y.data());

          /* Verify results */
          for (usize i = 0; i < n(); i++) {
            for (usize j = 0; j < g(); j++) {
              ASSERT_EQ(u32(y[i * g() + j]), u32(x[j * n() + i]))
                  << "at element " << i << ", group " << j;
            }
          }
        }
        */
    }
}
