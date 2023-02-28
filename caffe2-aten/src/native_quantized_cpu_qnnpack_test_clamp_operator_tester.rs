crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/clamp-operator-tester.h]

pub struct ClampOperatorTester {
    batch_size:    usize, // default = { 1 }
    channels:      usize, // default = { 1 }
    input_stride:  usize, // default = { 0 }
    output_stride: usize, // default = { 0 }
    qmin:          u8, // default = { 0 }
    qmax:          u8, // default = { 255 }
    iterations:    usize, // default = { 15 }
}

impl ClampOperatorTester {
    
    #[inline] pub fn channels(&mut self, channels: usize) -> &mut ClampOperatorTester {
        
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
    
    #[inline] pub fn input_stride(&mut self, input_stride: usize) -> &mut ClampOperatorTester {
        
        todo!();
        /*
            assert(inputStride != 0);
        this->inputStride_ = inputStride;
        return *this;
        */
    }
    
    #[inline] pub fn input_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->inputStride_ == 0) {
          return this->channels_;
        } else {
          assert(this->inputStride_ >= this->channels_);
          return this->inputStride_;
        }
        */
    }
    
    #[inline] pub fn output_stride(&mut self, output_stride: usize) -> &mut ClampOperatorTester {
        
        todo!();
        /*
            assert(outputStride != 0);
        this->outputStride_ = outputStride;
        return *this;
        */
    }
    
    #[inline] pub fn output_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->outputStride_ == 0) {
          return this->channels_;
        } else {
          assert(this->outputStride_ >= this->channels_);
          return this->outputStride_;
        }
        */
    }
    
    #[inline] pub fn batch_size(&mut self, batch_size: usize) -> &mut ClampOperatorTester {
        
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
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut ClampOperatorTester {
        
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
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut ClampOperatorTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut ClampOperatorTester {
        
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
    
    pub fn testu8(&self)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> input((batchSize() - 1) * inputStride() + channels());
        vector<u8> output(
            (batchSize() - 1) * outputStride() + channels());
        vector<u8> outputRef(batchSize() * channels());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(input.begin(), input.end(), ref(u8rng));
          fill(output.begin(), output.end(), 0xA5);

          /* Compute reference results */
          for (usize i = 0; i < batchSize(); i++) {
            for (usize c = 0; c < channels(); c++) {
              const u8 x = input[i * inputStride() + c];
              const u8 y = min(max(x, qmin()), qmax());
              outputRef[i * channels() + c] = y;
            }
          }

          /* Create, setup, run, and destroy Sigmoid operator */
          ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
          pytorch_qnnp_operator_t clampOp = nullptr;

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_create_clamp_nc_u8(
                  channels(), qmin(), qmax(), 0, &clampOp));
          ASSERT_NE(nullptr, clampOp);

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_setup_clamp_nc_u8(
                  clampOp,
                  batchSize(),
                  input.data(),
                  inputStride(),
                  output.data(),
                  outputStride()));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_run_operator(clampOp, nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success, pytorch_qnnp_delete_operator(clampOp));
          clampOp = nullptr;

          /* Verify results */
          for (usize i = 0; i < batchSize(); i++) {
            for (usize c = 0; c < channels(); c++) {
              ASSERT_LE(u32(output[i * channels() + c]), u32(qmax()))
                  << "at position " << i << ", batch size = " << batchSize()
                  << ", channels = " << channels();
              ASSERT_GE(u32(output[i * channels() + c]), u32(qmin()))
                  << "at position " << i << ", batch size = " << batchSize()
                  << ", channels = " << channels();
              ASSERT_EQ(
                  u32(outputRef[i * channels() + c]),
                  u32(output[i * outputStride() + c]))
                  << "at position " << i << ", batch size = " << batchSize()
                  << ", channels = " << channels() << ", qmin = " << qmin()
                  << ", qmax = " << qmax();
            }
          }
        }
        */
    }
}

 
