// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/maxpool-microkernel-tester.h]

pub struct MaxPoolMicrokernelTester {
    n:          usize, // default = { 1 }
    s:          usize, // default = { 1 }
    kh:         usize, // default = { 1 }
    kw:         usize, // default = { 1 }
    mr:         usize, // default = { 1 }
    qr:         usize, // default = { 1 }
    kc:         usize, // default = { 1 }
    kr:         usize, // default = { 1 }
    x_stride:   usize, // default = { 0 }
    y_stride:   usize, // default = { 0 }
    qmin:       u8, // default = { 0 }
    qmax:       u8, // default = { 255 }
    iterations: usize, // default = { 15 }
}

impl MaxPoolMicrokernelTester {

    
    #[inline] pub fn n(&mut self, n: usize) -> &mut MaxPoolMicrokernelTester {
        
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
    
    #[inline] pub fn s(&mut self, s: usize) -> &mut MaxPoolMicrokernelTester {
        
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
    
    #[inline] pub fn kh(&mut self, kh: usize) -> &mut MaxPoolMicrokernelTester {
        
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
    
    #[inline] pub fn kw(&mut self, kw: usize) -> &mut MaxPoolMicrokernelTester {
        
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
    
    #[inline] pub fn mr(&mut self, mr: usize) -> &mut MaxPoolMicrokernelTester {
        
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
    
    #[inline] pub fn qr(&mut self, qr: usize) -> &mut MaxPoolMicrokernelTester {
        
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
    
    #[inline] pub fn kc(&mut self, kc: usize) -> &mut MaxPoolMicrokernelTester {
        
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
    
    #[inline] pub fn kr(&mut self, kr: usize) -> &mut MaxPoolMicrokernelTester {
        
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
    
    #[inline] pub fn x_stride(&mut self, x_stride: usize) -> &mut MaxPoolMicrokernelTester {
        
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
    
    #[inline] pub fn y_stride(&mut self, y_stride: usize) -> &mut MaxPoolMicrokernelTester {
        
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
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut MaxPoolMicrokernelTester {
        
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
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut MaxPoolMicrokernelTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut MaxPoolMicrokernelTester {
        
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
    
    pub fn test(&self, u8maxpool: PyTorchU8MaxPoolUKernelFunction)  {
        
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
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(x.begin(), x.end(), ref(u8rng));
          fill(y.begin(), y.end(), 0xA5);

          for (usize i = 0; i < indirectX.size(); i++) {
            indirectX[i] = x.data() + i * xStride();
          }
          shuffle(indirectX.begin(), indirectX.end(), rng);

          /* Prepare quantization parameters */
          const union pytorch_qnnp_u8_clamping_params clampingParams =
              pytorch_qnnp_compute_u8_clamping_params(qmin(), qmax());

          /* Compute reference results */
          for (usize i = 0; i < n(); i++) {
            for (usize k = 0; k < kc(); k++) {
              u8 maxValue = 0;
              for (usize j = 0; j < ks(); j++) {
                maxValue = max(maxValue, indirectX[i * s() * kh() + j][k]);
              }
              maxValue = min(maxValue, qmax());
              maxValue = max(maxValue, qmin());
              yRef[i * kc() + k] = maxValue;
            }
          }

          /* Call optimized micro-kernel */
          u8maxpool(
              n(),
              ks(),
              kc(),
              indirectX.data(),
              y.data(),
              (kh() * s() - packedKs()) * sizeof(void*),
              (yStride() - kc()) * sizeof(u8),
              &clampingParams);

          /* Verify results */
          for (usize i = 0; i < n(); i++) {
            for (usize k = 0; k < kc(); k++) {
              ASSERT_EQ(
                  u32(yRef[i * kc() + k]), u32(y[i * yStride() + k]))
                  << "at pixel " << i << ", channel " << k << ", n = " << n()
                  << ", ks = " << kh() << "x" << kw() << " (" << ks()
                  << "), kc = " << kc();
            }
          }
        }
        */
    }
}



 
