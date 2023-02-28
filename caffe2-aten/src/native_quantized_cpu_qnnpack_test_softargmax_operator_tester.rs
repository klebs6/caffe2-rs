crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/softargmax-operator-tester.h]

pub struct SoftArgMaxOperatorTester {
    batch_size:       usize, // default = { 1 }
    channels:         usize, // default = { 1 }
    input_stride:     usize, // default = { 0 }
    output_stride:    usize, // default = { 0 }
    input_scale:      f32, // default = { 0.176080093 }
    input_zero_point: u8, // default = { 121 }
    iterations:       usize, // default = { 15 }
}

impl SoftArgMaxOperatorTester {
    
    #[inline] pub fn channels(&mut self, channels: usize) -> &mut SoftArgMaxOperatorTester {
        
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
    
    #[inline] pub fn input_stride(&mut self, input_stride: usize) -> &mut SoftArgMaxOperatorTester {
        
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
    
    #[inline] pub fn output_stride(&mut self, output_stride: usize) -> &mut SoftArgMaxOperatorTester {
        
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
    
    #[inline] pub fn batch_size(&mut self, batch_size: usize) -> &mut SoftArgMaxOperatorTester {
        
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
    
    #[inline] pub fn input_scale(&mut self, input_scale: f32) -> &mut SoftArgMaxOperatorTester {
        
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
    
    #[inline] pub fn input_zero_point(&mut self, input_zero_point: u8) -> &mut SoftArgMaxOperatorTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut SoftArgMaxOperatorTester {
        
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
            const i32 maxInput = *max_element(
                input.data() + i * inputStride(),
                input.data() + i * inputStride() + channels());
            float sumExp = 0.0f;
            for (usize c = 0; c < channels(); c++) {
              sumExp +=
                  exp((i32(input[i * inputStride() + c]) - maxInput) *
                      inputScale());
            }
            for (usize c = 0; c < channels(); c++) {
              outputRef[i * channels() + c] =
                  exp((i32(input[i * inputStride() + c]) - maxInput) *
                      inputScale()) /
                  (sumExp * outputScale());
              outputRef[i * channels() + c] =
                  min(outputRef[i * channels() + c], 255.0f);
            }
          }

          /* Create, setup, run, and destroy SoftArgMax operator */
          ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
          pytorch_qnnp_operator_t softArgMaxOp = nullptr;

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_create_softargmax_nc_q8(
                  channels(),
                  inputScale(),
                  outputZeroPoint(),
                  outputScale(),
                  0,
                  &softArgMaxOp));
          ASSERT_NE(nullptr, softArgMaxOp);

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_setup_softargmax_nc_q8(
                  softArgMaxOp,
                  batchSize(),
                  input.data(),
                  inputStride(),
                  output.data(),
                  outputStride()));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_run_operator(softArgMaxOp, nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_delete_operator(softArgMaxOp));
          softArgMaxOp = nullptr;

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
