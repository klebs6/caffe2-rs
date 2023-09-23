crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/avgpool-microkernel-tester.h]

pub struct AvgPoolMicrokernelTester {
    n:            usize, // default = { 1 }
    s:            usize, // default = { 1 }
    kh:           usize, // default = { 1 }
    kw:           usize, // default = { 1 }
    mr:           usize, // default = { 1 }
    qr:           usize, // default = { 1 }
    kc:           usize, // default = { 1 }
    kr:           usize, // default = { 1 }
    x_stride:     usize, // default = { 0 }
    y_stride:     usize, // default = { 0 }
    x_scale:      f32, // default = { 1.25f }
    y_scale:      f32, // default = { 0.75f }
    x_zero_point: u8, // default = { 121 }
    y_zero_point: u8, // default = { 133 }
    y_min:        u8, // default = { 0 }
    y_max:        u8, // default = { 255 }
    iterations:   usize, // default = { 15 }
}

impl AvgPoolMicrokernelTester {
    
    #[inline] pub fn n(&mut self, n: usize) -> &mut AvgPoolMicrokernelTester {
        
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
    
    #[inline] pub fn s(&mut self, s: usize) -> &mut AvgPoolMicrokernelTester {
        
        todo!();
        /*
            assert(s != 0);
        this->s_ = s;
        return *this;
        */
    }
    
    #[inline] pub fn s(&self) -> usize {
        
        todo!();
        /*
            return this->s_;
        */
    }
    
    #[inline] pub fn kh(&mut self, kh: usize) -> &mut AvgPoolMicrokernelTester {
        
        todo!();
        /*
            assert(kh != 0);
        this->kh_ = kh;
        return *this;
        */
    }
    
    #[inline] pub fn kh(&self) -> usize {
        
        todo!();
        /*
            return this->kh_;
        */
    }
    
    #[inline] pub fn kw(&mut self, kw: usize) -> &mut AvgPoolMicrokernelTester {
        
        todo!();
        /*
            assert(kw != 0);
        this->kw_ = kw;
        return *this;
        */
    }
    
    #[inline] pub fn kw(&self) -> usize {
        
        todo!();
        /*
            return this->kw_;
        */
    }
    
    #[inline] pub fn ks(&self) -> usize {
        
        todo!();
        /*
            return kh() * kw();
        */
    }
    
    #[inline] pub fn packed_ks(&self) -> usize {
        
        todo!();
        /*
            if (kc() < kr()) {
          return ks();
        } else if (ks() <= mr()) {
          return mr();
        } else {
          return (ks() - mr()) % qr() == 0
              ? ks()
              : ((ks() - mr()) / qr() + 1) * qr() + mr();
        }
        */
    }
    
    #[inline] pub fn mr(&mut self, mr: usize) -> &mut AvgPoolMicrokernelTester {
        
        todo!();
        /*
            assert(mr != 0);
        this->mr_ = mr;
        return *this;
        */
    }
    
    #[inline] pub fn mr(&self) -> usize {
        
        todo!();
        /*
            return this->mr_;
        */
    }
    
    #[inline] pub fn qr(&mut self, qr: usize) -> &mut AvgPoolMicrokernelTester {
        
        todo!();
        /*
            assert(qr != 0);
        this->qr_ = qr;
        return *this;
        */
    }
    
    #[inline] pub fn qr(&self) -> usize {
        
        todo!();
        /*
            return this->qr_;
        */
    }
    
    #[inline] pub fn kc(&mut self, kc: usize) -> &mut AvgPoolMicrokernelTester {
        
        todo!();
        /*
            assert(kc != 0);
        this->kc_ = kc;
        return *this;
        */
    }
    
    #[inline] pub fn kc(&self) -> usize {
        
        todo!();
        /*
            return this->kc_;
        */
    }
    
    #[inline] pub fn kr(&mut self, kr: usize) -> &mut AvgPoolMicrokernelTester {
        
        todo!();
        /*
            assert(kr != 0);
        this->kr_ = kr;
        return *this;
        */
    }
    
    #[inline] pub fn kr(&self) -> usize {
        
        todo!();
        /*
            return this->kr_;
        */
    }
    
    #[inline] pub fn packedn(&self) -> usize {
        
        todo!();
        /*
            return kc() % kr() == 0 ? kc() : (kc() / kr() + 1) * kr();
        */
    }
    
    #[inline] pub fn x_stride(&mut self, x_stride: usize) -> &mut AvgPoolMicrokernelTester {
        
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
          return kc();
        } else {
          assert(this->xStride_ >= kc());
          return this->xStride_;
        }
        */
    }
    
    #[inline] pub fn y_stride(&mut self, y_stride: usize) -> &mut AvgPoolMicrokernelTester {
        
        todo!();
        /*
            assert(yStride != 0);
        this->yStride_ = yStride;
        return *this;
        */
    }
    
    #[inline] pub fn y_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->yStride_ == 0) {
          return kc();
        } else {
          assert(this->yStride_ >= kc());
          return this->yStride_;
        }
        */
    }
    
    #[inline] pub fn x_scale(&mut self, x_scale: f32) -> &mut AvgPoolMicrokernelTester {
        
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
    
    #[inline] pub fn x_zero_point(&mut self, x_zero_point: u8) -> &mut AvgPoolMicrokernelTester {
        
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
    
    #[inline] pub fn y_scale(&mut self, y_scale: f32) -> &mut AvgPoolMicrokernelTester {
        
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
    
    #[inline] pub fn y_zero_point(&mut self, y_zero_point: u8) -> &mut AvgPoolMicrokernelTester {
        
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
    
    #[inline] pub fn y_min(&mut self, y_min: u8) -> &mut AvgPoolMicrokernelTester {
        
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
    
    #[inline] pub fn y_max(&mut self, y_max: u8) -> &mut AvgPoolMicrokernelTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut AvgPoolMicrokernelTester {
        
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
    
    pub fn test(&self, q8avgpool: PyTorchQ8AvgPoolUpUKernelFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<const u8*> indirectX(packedKs() + (n() * s() - 1) * kh());
        vector<u8> x((indirectX.size() - 1) * xStride() + kc());

        vector<u8> zero(kc());
        vector<u8> y((n() - 1) * yStride() + kc());
        vector<u8> yRef(n() * kc());
        vector<float> yFP(n() * kc());
        vector<i32> yAcc(n() * kc());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(x.begin(), x.end(), ref(u8rng));
          fill(y.begin(), y.end(), 0xA5);

          for (usize i = 0; i < indirectX.size(); i++) {
            indirectX[i] = x.data() + i * xStride();
          }
          shuffle(indirectX.begin(), indirectX.end(), rng);

          /* Prepare quantization parameters */
          const union pytorch_qnnp_avgpool_quantization_params quantizationParams =
              pytorch_qnnp_compute_avgpool_quantization_params(
                  -i32(xZeroPoint()) * i32(ks()),
                  xScale() / (yScale() * float(ks())),
                  yZeroPoint(),
                  yMin(),
                  yMax());
          const union pytorch_qnnp_avgpool_quantization_params
              scalarQuantizationParams =
                  pytorch_qnnp_compute_scalar_avgpool_quantization_params(
                      -i32(xZeroPoint()) * i32(ks()),
                      xScale() / (yScale() * float(ks())),
                      yZeroPoint(),
                      yMin(),
                      yMax());

          /* Compute reference results */
          for (usize i = 0; i < n(); i++) {
            for (usize k = 0; k < kc(); k++) {
              i32 acc = scalarQuantizationParams.scalar.bias;
              for (usize j = 0; j < ks(); j++) {
                acc += indirectX[i * s() * kh() + j][k];
              }
              yAcc[i * kc() + k] = acc;
              yRef[i * kc() + k] =
                  pytorch_qnnp_avgpool_quantize(acc, scalarQuantizationParams);
              yFP[i * kc() + k] =
                  float(acc) * (xScale() / (yScale() * float(ks()))) +
                  float(yZeroPoint());
              yFP[i * kc() + k] = min<float>(yFP[i * kc() + k], float(yMax()));
              yFP[i * kc() + k] = max<float>(yFP[i * kc() + k], float(yMin()));
            }
          }

          /* Call optimized micro-kernel */
          q8avgpool(
              n(),
              ks(),
              kc(),
              indirectX.data(),
              zero.data(),
              y.data(),
              kh() * s() * sizeof(void*),
              (yStride() - kc()) * sizeof(u8),
              &quantizationParams);

          /* Verify results */
          for (usize i = 0; i < n(); i++) {
            for (usize k = 0; k < kc(); k++) {
              ASSERT_LE(u32(y[i * yStride() + k]), u32(yMax()))
                  << "at pixel " << i << ", channel " << k << ", n = " << n()
                  << ", kc = " << kc();
              ASSERT_GE(u32(y[i * yStride() + k]), u32(yMin()))
                  << "at pixel " << i << ", channel " << k << ", n = " << n()
                  << ", kc = " << kc();
              ASSERT_NEAR(
                  float(i32(y[i * yStride() + k])), yFP[i * kc() + k], 0.5f)
                  << "at pixel " << i << ", channel " << k << ", n = " << n()
                  << ", ks = " << kh() << "x" << kw() << " (" << ks()
                  << "), kc = " << kc() << ", acc = " << yAcc[i * kc() + k];
              ASSERT_EQ(
                  u32(yRef[i * kc() + k]), u32(y[i * yStride() + k]))
                  << "at pixel " << i << ", channel " << k << ", n = " << n()
                  << ", ks = " << kh() << "x" << kw() << " (" << ks()
                  << "), kc = " << kc() << ", acc = " << yAcc[i * kc() + k];
            }
          }
        }
        */
    }
    
    pub fn test(&self, q8avgpool: PyTorchQ8AvgPoolMpUKernelFunction)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<const u8*> indirectX(packedKs() + (n() * s() - 1) * kh());
        vector<u8> x((indirectX.size() - 1) * xStride() + kc());
        vector<i32, AlignedAllocator<i32, 16>> mpAcc(packedN());

        vector<u8> zero(kc());
        vector<u8> y((n() - 1) * yStride() + kc());
        vector<u8> yRef(n() * kc());
        vector<float> yFP(n() * kc());
        vector<i32> yAcc(n() * kc());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(x.begin(), x.end(), ref(u8rng));
          fill(y.begin(), y.end(), 0xA5);

          for (usize i = 0; i < indirectX.size(); i++) {
            indirectX[i] = x.data() + i * xStride();
          }
          shuffle(indirectX.begin(), indirectX.end(), rng);

          /* Prepare quantization parameters */
          const union pytorch_qnnp_avgpool_quantization_params quantizationParams =
              pytorch_qnnp_compute_avgpool_quantization_params(
                  -i32(xZeroPoint()) * i32(ks()),
                  xScale() / (yScale() * float(ks())),
                  yZeroPoint(),
                  yMin(),
                  yMax());
          const union pytorch_qnnp_avgpool_quantization_params
              scalarQuantizationParams =
                  pytorch_qnnp_compute_scalar_avgpool_quantization_params(
                      -i32(xZeroPoint()) * i32(ks()),
                      xScale() / (yScale() * float(ks())),
                      yZeroPoint(),
                      yMin(),
                      yMax());

          /* Compute reference results */
          for (usize i = 0; i < n(); i++) {
            for (usize k = 0; k < kc(); k++) {
              i32 acc = scalarQuantizationParams.scalar.bias;
              for (usize j = 0; j < ks(); j++) {
                acc += indirectX[i * s() * kh() + j][k];
              }
              yAcc[i * kc() + k] = acc;
              yRef[i * kc() + k] =
                  pytorch_qnnp_avgpool_quantize(acc, scalarQuantizationParams);
              yFP[i * kc() + k] =
                  float(acc) * (xScale() / (yScale() * float(ks()))) +
                  float(yZeroPoint());
              yFP[i * kc() + k] = min<float>(yFP[i * kc() + k], float(yMax()));
              yFP[i * kc() + k] = max<float>(yFP[i * kc() + k], float(yMin()));
            }
          }

          /* Call optimized micro-kernel */
          q8avgpool(
              n(),
              ks(),
              kc(),
              indirectX.data(),
              zero.data(),
              mpAcc.data(),
              y.data(),
              (kh() * s() - (packedKs() - qr())) * sizeof(void*),
              (yStride() - kc()) * sizeof(u8),
              &quantizationParams);

          /* Verify results */
          for (usize i = 0; i < n(); i++) {
            for (usize k = 0; k < kc(); k++) {
              ASSERT_LE(u32(y[i * yStride() + k]), u32(yMax()))
                  << "at pixel " << i << ", channel " << k << ", n = " << n()
                  << ", kc = " << kc();
              ASSERT_GE(u32(y[i * yStride() + k]), u32(yMin()))
                  << "at pixel " << i << ", channel " << k << ", n = " << n()
                  << ", kc = " << kc();
              ASSERT_NEAR(
                  float(i32(y[i * yStride() + k])), yFP[i * kc() + k], 0.5f)
                  << "at pixel " << i << ", channel " << k << ", n = " << n()
                  << ", ks = " << kh() << "x" << kw() << " (" << ks()
                  << "), kc = " << kc() << ", acc = " << yAcc[i * kc() + k];
              ASSERT_EQ(
                  u32(yRef[i * kc() + k]), u32(y[i * yStride() + k]))
                  << "at pixel " << i << ", channel " << k << ", n = " << n()
                  << ", ks = " << kh() << "x" << kw() << " (" << ks()
                  << "), kc = " << kc() << ", acc = " << yAcc[i * kc() + k];
            }
          }
        }
        */
    }
}
