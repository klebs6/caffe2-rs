crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/sigmoid-operator-tester.h]

pub struct SigmoidOperatorTester {
    batch_size:       usize, // default = { 1 }
    channels:         usize, // default = { 1 }
    input_stride:     usize, // default = { 0 }
    output_stride:    usize, // default = { 0 }
    input_scale:      f32, // default = { 0.75f }
    input_zero_point: u8, // default = { 121 }
    qmin:             u8, // default = { 0 }
    qmax:             u8, // default = { 255 }
    iterations:       usize, // default = { 15 }
}

impl SigmoidOperatorTester {

    
    #[inline] pub fn channels(&mut self, channels: usize) -> &mut SigmoidOperatorTester {
        
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
    
    #[inline] pub fn input_stride(&mut self, input_stride: usize) -> &mut SigmoidOperatorTester {
        
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
    
    #[inline] pub fn output_stride(&mut self, output_stride: usize) -> &mut SigmoidOperatorTester {
        
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
    
    #[inline] pub fn batch_size(&mut self, batch_size: usize) -> &mut SigmoidOperatorTester {
        
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
    
    #[inline] pub fn input_scale(&mut self, input_scale: f32) -> &mut SigmoidOperatorTester {
        
        todo!();
        /*
            assert(inputScale > 0.0f);
        assert(isnormal(inputScale));
        this->inputScale_ = inputScale;
        return *this;
        */
    }
    
    #[inline] pub fn input_scale(&self) -> f32 {
        
        todo!();
        /*
            return this->inputScale_;
        */
    }
    
    #[inline] pub fn input_zero_point(&mut self, input_zero_point: u8) -> &mut SigmoidOperatorTester {
        
        todo!();
        /*
            this->inputZeroPoint_ = inputZeroPoint;
        return *this;
        */
    }
    
    #[inline] pub fn input_zero_point(&self) -> u8 {
        
        todo!();
        /*
            return this->inputZeroPoint_;
        */
    }
    
    #[inline] pub fn output_scale(&self) -> f32 {
        
        todo!();
        /*
            return 1.0f / 256.0f;
        */
    }
    
    #[inline] pub fn output_zero_point(&self) -> u8 {
        
        todo!();
        /*
            return 0;
        */
    }
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut SigmoidOperatorTester {
        
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
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut SigmoidOperatorTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut SigmoidOperatorTester {
        
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

        vector<u8> input((batchSize() - 1) * inputStride() + channels());
        vector<u8> output(
            (batchSize() - 1) * outputStride() + channels());
        vector<float> outputRef(batchSize() * channels());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(input.begin(), input.end(), ref(u8rng));
          fill(output.begin(), output.end(), 0xA5);

          /* Compute reference results */
          for (usize i = 0; i < batchSize(); i++) {
            for (usize c = 0; c < channels(); c++) {
              const float x = inputScale() *
                  (i32(input[i * inputStride() + c]) -
                   i32(inputZeroPoint()));
              const float sigmoidX = 1.0f / (1.0f + exp(-x));
              const float scaledSigmoidX = sigmoidX / outputScale();
              float y = scaledSigmoidX;
              y = min<float>(y, i32(qmax()) - i32(outputZeroPoint()));
              y = max<float>(y, i32(qmin()) - i32(outputZeroPoint()));
              outputRef[i * channels() + c] = y + i32(outputZeroPoint());
            }
          }

          /* Create, setup, run, and destroy Sigmoid operator */
          ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
          pytorch_qnnp_operator_t sigmoidOp = nullptr;

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_create_sigmoid_nc_q8(
                  channels(),
                  inputZeroPoint(),
                  inputScale(),
                  outputZeroPoint(),
                  outputScale(),
                  qmin(),
                  qmax(),
                  0,
                  &sigmoidOp));
          ASSERT_NE(nullptr, sigmoidOp);

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_setup_sigmoid_nc_q8(
                  sigmoidOp,
                  batchSize(),
                  input.data(),
                  inputStride(),
                  output.data(),
                  outputStride()));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_run_operator(sigmoidOp, nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success, pytorch_qnnp_delete_operator(sigmoidOp));
          sigmoidOp = nullptr;

          /* Verify results */
          for (usize i = 0; i < batchSize(); i++) {
            for (usize c = 0; c < channels(); c++) {
              ASSERT_NEAR(
                  float(i32(output[i * outputStride() + c])),
                  outputRef[i * channels() + c],
                  0.6f);
            }
          }
        }
        */
    }
}
