crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/add-operator-tester.h]

pub struct AddOperatorTester {
    batch_size:   usize, // default = { 1 }
    channels:     usize, // default = { 1 }
    a_stride:     usize, // default = { 0 }
    b_stride:     usize, // default = { 0 }
    y_stride:     usize, // default = { 0 }
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

impl AddOperatorTester {
    
    #[inline] pub fn channels(&mut self, channels: usize) -> &mut AddOperatorTester {
        
        todo!();
        /*
            assert(channels != 0);
        this->channels_ = channels;
        return *this;
        */
    }
    
    #[inline] pub fn channels(&self) -> usize {
        
        todo!();
        /*
            return this->channels_;
        */
    }
    
    #[inline] pub fn a_stride(&mut self, a_stride: usize) -> &mut AddOperatorTester {
        
        todo!();
        /*
            assert(aStride != 0);
        this->aStride_ = aStride;
        return *this;
        */
    }
    
    #[inline] pub fn a_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->aStride_ == 0) {
          return this->channels_;
        } else {
          assert(this->aStride_ >= this->channels_);
          return this->aStride_;
        }
        */
    }
    
    #[inline] pub fn b_stride(&mut self, b_stride: usize) -> &mut AddOperatorTester {
        
        todo!();
        /*
            assert(bStride != 0);
        this->bStride_ = bStride;
        return *this;
        */
    }
    
    #[inline] pub fn b_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->bStride_ == 0) {
          return this->channels_;
        } else {
          assert(this->bStride_ >= this->channels_);
          return this->bStride_;
        }
        */
    }
    
    #[inline] pub fn y_stride(&mut self, y_stride: usize) -> &mut AddOperatorTester {
        
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
          return this->channels_;
        } else {
          assert(this->yStride_ >= this->channels_);
          return this->yStride_;
        }
        */
    }
    
    #[inline] pub fn batch_size(&mut self, batch_size: usize) -> &mut AddOperatorTester {
        
        todo!();
        /*
            this->batchSize_ = batchSize;
        return *this;
        */
    }
    
    #[inline] pub fn batch_size(&self) -> usize {
        
        todo!();
        /*
            return this->batchSize_;
        */
    }
    
    #[inline] pub fn a_scale(&mut self, a_scale: f32) -> &mut AddOperatorTester {
        
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
    
    #[inline] pub fn a_zero_point(&mut self, a_zero_point: u8) -> &mut AddOperatorTester {
        
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
    
    #[inline] pub fn b_scale(&mut self, b_scale: f32) -> &mut AddOperatorTester {
        
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
    
    #[inline] pub fn b_zero_point(&mut self, b_zero_point: u8) -> &mut AddOperatorTester {
        
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
    
    #[inline] pub fn y_scale(&mut self, y_scale: f32) -> &mut AddOperatorTester {
        
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
    
    #[inline] pub fn y_zero_point(&mut self, y_zero_point: u8) -> &mut AddOperatorTester {
        
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
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut AddOperatorTester {
        
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
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut AddOperatorTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut AddOperatorTester {
        
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
    
    pub fn testq8(&self)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> a((batchSize() - 1) * aStride() + channels());
        vector<u8> b((batchSize() - 1) * bStride() + channels());
        vector<u8> y((batchSize() - 1) * yStride() + channels());
        vector<float> yRef(batchSize() * channels());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(a.begin(), a.end(), ref(u8rng));
          generate(b.begin(), b.end(), ref(u8rng));
          fill(y.begin(), y.end(), 0xA5);

          if (batchSize() * channels() > 3) {
            ASSERT_NE(
                *max_element(a.cbegin(), a.cend()),
                *min_element(a.cbegin(), a.cend()));
            ASSERT_NE(
                *max_element(b.cbegin(), b.cend()),
                *min_element(b.cbegin(), b.cend()));
          }

          /* Compute reference results */
          for (usize i = 0; i < batchSize(); i++) {
            for (usize c = 0; c < channels(); c++) {
              yRef[i * channels() + c] = float(yZeroPoint()) +
                  float(i32(a[i * aStride() + c]) - i32(aZeroPoint())) *
                      (aScale() / yScale()) +
                  float(i32(b[i * bStride() + c]) - i32(bZeroPoint())) *
                      (bScale() / yScale());
              yRef[i * channels() + c] =
                  min<float>(yRef[i * channels() + c], float(qmax()));
              yRef[i * channels() + c] =
                  max<float>(yRef[i * channels() + c], float(qmin()));
            }
          }

          /* Create, setup, run, and destroy Add operator */
          ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
          pytorch_qnnp_operator_t add_op = nullptr;

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_create_add_nc_q8(
                  channels(),
                  aZeroPoint(),
                  aScale(),
                  bZeroPoint(),
                  bScale(),
                  yZeroPoint(),
                  yScale(),
                  qmin(),
                  qmax(),
                  0,
                  &add_op));
          ASSERT_NE(nullptr, add_op);

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_setup_add_nc_q8(
                  add_op,
                  batchSize(),
                  a.data(),
                  aStride(),
                  b.data(),
                  bStride(),
                  y.data(),
                  yStride()));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_run_operator(add_op, nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success, pytorch_qnnp_delete_operator(add_op));
          add_op = nullptr;

          /* Verify results */
          for (usize i = 0; i < batchSize(); i++) {
            for (usize c = 0; c < channels(); c++) {
              ASSERT_LE(u32(y[i * yStride() + c]), u32(qmax()));
              ASSERT_GE(u32(y[i * yStride() + c]), u32(qmin()));
              ASSERT_NEAR(
                  float(i32(y[i * yStride() + c])),
                  yRef[i * channels() + c],
                  0.6f);
            }
          }
        }
        */
    }
}
