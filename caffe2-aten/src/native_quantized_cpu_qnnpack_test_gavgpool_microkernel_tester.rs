// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/gavgpool-microkernel-tester.h]

pub struct GAvgPoolMicrokernelTester {
    m:            usize, // default = { 1 }
    n:            usize, // default = { 1 }
    nr:           usize, // default = { 1 }
    x_stride:     usize, // default = { 0 }
    x_scale:      f32, // default = { 1.25f }
    y_scale:      f32, // default = { 0.75f }
    x_zero_point: u8, // default = { 121 }
    y_zero_point: u8, // default = { 133 }
    y_min:        u8, // default = { 0 }
    y_max:        u8, // default = { 255 }
    iterations:   usize, // default = { 15 }
}

impl GAvgPoolMicrokernelTester {
    
    #[inline] pub fn m(&mut self, m: usize) -> &mut GAvgPoolMicrokernelTester {
        
        todo!();
        /*
            assert(m != 0);
        this->m_ = m;
        return *this;
        */
    }
    
    #[inline] pub fn m(&self) -> usize {
        
        todo!();
        /*
            return this->m_;
        */
    }
    
    #[inline] pub fn n(&mut self, n: usize) -> &mut GAvgPoolMicrokernelTester {
        
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
    
    #[inline] pub fn nr(&mut self, nr: usize) -> &mut GAvgPoolMicrokernelTester {
        
        todo!();
        /*
            assert(nr != 0);
        this->nr_ = nr;
        return *this;
        */
    }
    
    #[inline] pub fn nr(&self) -> usize {
        
        todo!();
        /*
            return this->nr_;
        */
    }
    
    #[inline] pub fn packedn(&self) -> usize {
        
        todo!();
        /*
            return n() % nr() == 0 ? n() : (n() / nr() + 1) * nr();
        */
    }
    
    #[inline] pub fn x_stride(&mut self, x_stride: usize) -> &mut GAvgPoolMicrokernelTester {
        
        todo!();
        /*
            assert(xStride != 0);
        this->xStride_ = xStride;
        return *this;
        */
    }
    
    #[inline] pub fn x_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->xStride_ == 0) {
          return n();
        } else {
          assert(this->xStride_ >= n());
          return this->xStride_;
        }
        */
    }
    
    #[inline] pub fn x_scale(&mut self, x_scale: f32) -> &mut GAvgPoolMicrokernelTester {
        
        todo!();
        /*
            assert(xScale > 0.0f);
        assert(isnormal(xScale));
        this->xScale_ = xScale;
        return *this;
        */
    }
    
    #[inline] pub fn x_scale(&self) -> f32 {
        
        todo!();
        /*
            return this->xScale_;
        */
    }
    
    #[inline] pub fn x_zero_point(&mut self, x_zero_point: u8) -> &mut GAvgPoolMicrokernelTester {
        
        todo!();
        /*
            this->xZeroPoint_ = xZeroPoint;
        return *this;
        */
    }
    
    #[inline] pub fn x_zero_point(&self) -> u8 {
        
        todo!();
        /*
            return this->xZeroPoint_;
        */
    }
    
    #[inline] pub fn y_scale(&mut self, y_scale: f32) -> &mut GAvgPoolMicrokernelTester {
        
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
    
    #[inline] pub fn y_zero_point(&mut self, y_zero_point: u8) -> &mut GAvgPoolMicrokernelTester {
        
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
    
    #[inline] pub fn y_min(&mut self, y_min: u8) -> &mut GAvgPoolMicrokernelTester {
        
        todo!();
        /*
            this->yMin_ = yMin;
        return *this;
        */
    }
    
    #[inline] pub fn y_min(&self) -> u8 {
        
        todo!();
        /*
            return this->yMin_;
        */
    }
    
    #[inline] pub fn y_max(&mut self, y_max: u8) -> &mut GAvgPoolMicrokernelTester {
        
        todo!();
        /*
            this->yMax_ = yMax;
        return *this;
        */
    }
    
    #[inline] pub fn y_max(&self) -> u8 {
        
        todo!();
        /*
            return this->yMax_;
        */
    }
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut GAvgPoolMicrokernelTester {
        
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
    
    pub fn test(&self, q8gavgpool: PyTorchQ8GAvgPoolUpUKernelFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> x((m() - 1) * xStride() + n());
        vector<u8> zero(n());
        vector<u8> y(n());
        vector<u8> yRef(n());
        vector<float> yFP(n());
        vector<i32> yAcc(n());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(x.begin(), x.end(), ref(u8rng));
          fill(y.begin(), y.end(), 0xA5);

          /* Prepare quantization parameters */
          const union pytorch_qnnp_avgpool_quantization_params quantizationParams =
              pytorch_qnnp_compute_avgpool_quantization_params(
                  -i32(xZeroPoint()) * i32(m()),
                  xScale() / (yScale() * float(m())),
                  yZeroPoint(),
                  yMin(),
                  yMax());
          const union pytorch_qnnp_avgpool_quantization_params
              scalarQuantizationParams =
                  pytorch_qnnp_compute_scalar_avgpool_quantization_params(
                      -i32(xZeroPoint()) * i32(m()),
                      xScale() / (yScale() * float(m())),
                      yZeroPoint(),
                      yMin(),
                      yMax());

          /* Compute reference results */
          for (usize j = 0; j < n(); j++) {
            i32 acc = scalarQuantizationParams.scalar.bias;
            for (usize i = 0; i < m(); i++) {
              acc += x[i * xStride() + j];
            }
            yAcc[j] = acc;
            yRef[j] = pytorch_qnnp_avgpool_quantize(acc, scalarQuantizationParams);
            yFP[j] = float(acc) * (xScale() / (yScale() * float(m()))) +
                float(yZeroPoint());
            yFP[j] = min<float>(yFP[j], float(yMax()));
            yFP[j] = max<float>(yFP[j], float(yMin()));
          }

          /* Call optimized micro-kernel */
          q8gavgpool(
              m(),
              n(),
              x.data(),
              xStride() * sizeof(u8),
              zero.data(),
              y.data(),
              &quantizationParams);

          /* Verify results */
          for (usize i = 0; i < n(); i++) {
            ASSERT_LE(u32(y[i]), u32(yMax()))
                << "at position " << i << ", m = " << m() << ", n = " << n();
            ASSERT_GE(u32(y[i]), u32(yMin()))
                << "at position " << i << ", m = " << m() << ", n = " << n();
            ASSERT_NEAR(float(i32(y[i])), yFP[i], 0.5f)
                << "at position " << i << ", m = " << m() << ", n = " << n()
                << ", acc = " << yAcc[i];
            ASSERT_EQ(u32(yRef[i]), u32(y[i]))
                << "at position " << i << ", m = " << m() << ", n = " << n()
                << ", acc = " << yAcc[i];
          }
        }
        */
    }
    
    pub fn test(&self, q8gavgpool: PyTorchQ8GAvgPoolMpUKernelFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> x((m() - 1) * xStride() + n());
        vector<i32, AlignedAllocator<i32, 16>> mpAcc(packedN());
        vector<u8> zero(n());
        vector<u8> y(n());
        vector<u8> yRef(n());
        vector<float> yFP(n());
        vector<i32> yAcc(n());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(x.begin(), x.end(), ref(u8rng));
          fill(y.begin(), y.end(), 0xA5);

          /* Prepare quantization parameters */
          const union pytorch_qnnp_avgpool_quantization_params quantizationParams =
              pytorch_qnnp_compute_avgpool_quantization_params(
                  -i32(xZeroPoint()) * i32(m()),
                  xScale() / (yScale() * float(m())),
                  yZeroPoint(),
                  yMin(),
                  yMax());
          const union pytorch_qnnp_avgpool_quantization_params
              scalarQuantizationParams =
                  pytorch_qnnp_compute_scalar_avgpool_quantization_params(
                      -i32(xZeroPoint()) * i32(m()),
                      xScale() / (yScale() * float(m())),
                      yZeroPoint(),
                      yMin(),
                      yMax());

          /* Compute reference results */
          for (usize j = 0; j < n(); j++) {
            i32 acc = scalarQuantizationParams.scalar.bias;
            for (usize i = 0; i < m(); i++) {
              acc += x[i * xStride() + j];
            }

            yAcc[j] = acc;
            yRef[j] = pytorch_qnnp_avgpool_quantize(acc, scalarQuantizationParams);
            yFP[j] = float(acc) * (xScale() / (yScale() * float(m()))) +
                float(yZeroPoint());
            yFP[j] = min<float>(yFP[j], float(yMax()));
            yFP[j] = max<float>(yFP[j], float(yMin()));
          }

          /* Call optimized micro-kernel */
          q8gavgpool(
              m(),
              n(),
              x.data(),
              xStride() * sizeof(u8),
              zero.data(),
              mpAcc.data(),
              y.data(),
              &quantizationParams);

          /* Verify results */
          for (usize i = 0; i < n(); i++) {
            ASSERT_LE(u32(y[i]), u32(yMax()))
                << "at position " << i << ", m = " << m() << ", n = " << n();
            ASSERT_GE(u32(y[i]), u32(yMin()))
                << "at position " << i << ", m = " << m() << ", n = " << n();
            ASSERT_NEAR(float(i32(y[i])), yFP[i], 0.5f)
                << "at position " << i << ", m = " << m() << ", n = " << n()
                << ", acc = " << yAcc[i];
            ASSERT_EQ(u32(yRef[i]), u32(y[i]))
                << "at position " << i << ", m = " << m() << ", n = " << n()
                << ", acc = " << yAcc[i];
          }
        }
        */
    }
}
