crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/clamp-microkernel-tester.h]

pub struct ClampMicrokernelTester {
    n:          usize, // default = { 1 }
    inplace:    bool, // default = { false }
    qmin:       u8, // default = { 0 }
    qmax:       u8, // default = { 255 }
    iterations: usize, // default = { 15 }
}

impl ClampMicrokernelTester {

    
    #[inline] pub fn n(&mut self, n: usize) -> &mut ClampMicrokernelTester {
        
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
    
    #[inline] pub fn inplace(&mut self, inplace: bool) -> &mut ClampMicrokernelTester {
        
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
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut ClampMicrokernelTester {
        
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
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut ClampMicrokernelTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut ClampMicrokernelTester {
        
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
    
    pub fn test(&self, u8clamp: PyTorchU8ClampUKernelFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> x(n());
        vector<u8> y(n());
        vector<u8> yRef(n());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(x.begin(), x.end(), ref(u8rng));
          if (inplace()) {
            generate(y.begin(), y.end(), ref(u8rng));
          } else {
            fill(y.begin(), y.end(), 0xA5);
          }
          const u8* xData = inplace() ? y.data() : x.data();

          /* Prepare clamping parameters */
          const union pytorch_qnnp_u8_clamping_params clampingParams =
              pytorch_qnnp_compute_u8_clamping_params(qmin(), qmax());

          /* Compute reference results */
          for (usize i = 0; i < n(); i++) {
            yRef[i] = max(min(xData[i], qmax()), qmin());
          }

          /* Call optimized micro-kernel */
          u8clamp(n(), xData, y.data(), &clampingParams);

          /* Verify results */
          for (usize i = 0; i < n(); i++) {
            ASSERT_LE(u32(y[i]), u32(qmax()))
                << "at position " << i << ", n = " << n();
            ASSERT_GE(u32(y[i]), u32(qmin()))
                << "at position " << i << ", n = " << n();
            ASSERT_EQ(u32(yRef[i]), u32(y[i]))
                << "at position " << i << ", n = " << n() << ", qmin = " << qmin()
                << ", qmax = " << qmax();
          }
        }
        */
    }
}
