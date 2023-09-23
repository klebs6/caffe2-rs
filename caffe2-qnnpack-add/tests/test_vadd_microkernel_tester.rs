crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/vadd-microkernel-tester.h]

pub struct VAddMicrokernelTester {
    n:            usize, // default = { 1 }
    inplacea:     bool, // default = { false }
    inplaceb:     bool, // default = { false }
    a_scale:      f32, // default = { 0.75f }
    b_scale:      f32, // default = { 1.25f }
    y_scale:      f32, // default = { 0.96875f }
    a_zero_point: u8, // default = { 121 }
    b_zero_point: u8, // default = { 127 }
    y_zero_point: u8, // default = { 133 }
    qmin:         u8, // default = { 0 }
    qmax:         u8, // default = { 255 }
    iterations:   usize, // default = { 15 }
}

impl VAddMicrokernelTester {

    
    #[inline] pub fn n(&mut self, n: usize) -> &mut VAddMicrokernelTester {
        
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
    
    #[inline] pub fn inplacea(&mut self, inplacea: bool) -> &mut VAddMicrokernelTester {
        
        todo!();
        /*
            this->inplaceA_ = inplaceA;
        return *this;
        */
    }
    
    #[inline] pub fn inplacea(&self) -> bool {
        
        todo!();
        /*
            return this->inplaceA_;
        */
    }
    
    #[inline] pub fn inplaceb(&mut self, inplaceb: bool) -> &mut VAddMicrokernelTester {
        
        todo!();
        /*
            this->inplaceB_ = inplaceB;
        return *this;
        */
    }
    
    #[inline] pub fn inplaceb(&self) -> bool {
        
        todo!();
        /*
            return this->inplaceB_;
        */
    }
    
    #[inline] pub fn a_scale(&mut self, a_scale: f32) -> &mut VAddMicrokernelTester {
        
        todo!();
        /*
            assert(aScale > 0.0f);
        assert(isnormal(aScale));
        this->aScale_ = aScale;
        return *this;
        */
    }
    
    #[inline] pub fn a_scale(&self) -> f32 {
        
        todo!();
        /*
            return this->aScale_;
        */
    }
    
    #[inline] pub fn a_zero_point(&mut self, a_zero_point: u8) -> &mut VAddMicrokernelTester {
        
        todo!();
        /*
            this->aZeroPoint_ = aZeroPoint;
        return *this;
        */
    }
    
    #[inline] pub fn a_zero_point(&self) -> u8 {
        
        todo!();
        /*
            return this->aZeroPoint_;
        */
    }
    
    #[inline] pub fn b_scale(&mut self, b_scale: f32) -> &mut VAddMicrokernelTester {
        
        todo!();
        /*
            assert(bScale > 0.0f);
        assert(isnormal(bScale));
        this->bScale_ = bScale;
        return *this;
        */
    }
    
    #[inline] pub fn b_scale(&self) -> f32 {
        
        todo!();
        /*
            return this->bScale_;
        */
    }
    
    #[inline] pub fn b_zero_point(&mut self, b_zero_point: u8) -> &mut VAddMicrokernelTester {
        
        todo!();
        /*
            this->bZeroPoint_ = bZeroPoint;
        return *this;
        */
    }
    
    #[inline] pub fn b_zero_point(&self) -> u8 {
        
        todo!();
        /*
            return this->bZeroPoint_;
        */
    }
    
    #[inline] pub fn y_scale(&mut self, y_scale: f32) -> &mut VAddMicrokernelTester {
        
        todo!();
        /*
            assert(yScale > 0.0f);
        assert(isnormal(yScale));
        this->yScale_ = yScale;
        return *this;
        */
    }
    
    #[inline] pub fn y_scale(&self) -> f32 {
        
        todo!();
        /*
            return this->yScale_;
        */
    }
    
    #[inline] pub fn y_zero_point(&mut self, y_zero_point: u8) -> &mut VAddMicrokernelTester {
        
        todo!();
        /*
            this->yZeroPoint_ = yZeroPoint;
        return *this;
        */
    }
    
    #[inline] pub fn y_zero_point(&self) -> u8 {
        
        todo!();
        /*
            return this->yZeroPoint_;
        */
    }
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut VAddMicrokernelTester {
        
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
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut VAddMicrokernelTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut VAddMicrokernelTester {
        
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
    
    pub fn test(&self, q8vadd: PyTorchQ8VAddUKernelFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> a(n());
        vector<u8> b(n());
        vector<u8> y(n());
        vector<float> yFP(n());
        vector<u8> yRef(n());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(a.begin(), a.end(), ref(u8rng));
          generate(b.begin(), b.end(), ref(u8rng));
          if (inplaceA() || inplaceB()) {
            generate(y.begin(), y.end(), ref(u8rng));
          } else {
            fill(y.begin(), y.end(), 0xA5);
          }
          const u8* aData = inplaceA() ? y.data() : a.data();
          const u8* bData = inplaceB() ? y.data() : b.data();

          /* Prepare quantization parameters */
          const union pytorch_qnnp_add_quantization_params quantizationParams =
              pytorch_qnnp_compute_add_quantization_params(
                  aZeroPoint(),
                  bZeroPoint(),
                  yZeroPoint(),
                  aScale() / yScale(),
                  bScale() / yScale(),
                  qmin(),
                  qmax());
          const union pytorch_qnnp_add_quantization_params
              scalarQuantizationParams =
                  pytorch_qnnp_compute_scalar_add_quantization_params(
                      aZeroPoint(),
                      bZeroPoint(),
                      yZeroPoint(),
                      aScale() / yScale(),
                      bScale() / yScale(),
                      qmin(),
                      qmax());

          /* Compute reference results */
          for (usize i = 0; i < n(); i++) {
            yFP[i] = float(yZeroPoint()) +
                float(i32(aData[i]) - i32(aZeroPoint())) *
                    (aScale() / yScale()) +
                float(i32(bData[i]) - i32(bZeroPoint())) *
                    (bScale() / yScale());
            yFP[i] = min<float>(yFP[i], float(qmax()));
            yFP[i] = max<float>(yFP[i], float(qmin()));
            yRef[i] = pytorch_qnnp_add_quantize(
                aData[i], bData[i], scalarQuantizationParams);
          }

          /* Call optimized micro-kernel */
          q8vadd(n(), aData, bData, y.data(), &quantizationParams);

          /* Verify results */
          for (usize i = 0; i < n(); i++) {
            ASSERT_LE(u32(y[i]), u32(qmax()))
                << "at " << i << ", n = " << n();
            ASSERT_GE(u32(y[i]), u32(qmin()))
                << "at " << i << ", n = " << n();
            ASSERT_NEAR(float(i32(y[i])), yFP[i], 0.6f)
                << "at " << i << ", n = " << n();
            ASSERT_EQ(u32(yRef[i]), u32(y[i]))
                << "at " << i << ", n = " << n();
          }
        }
        */
    }
}
