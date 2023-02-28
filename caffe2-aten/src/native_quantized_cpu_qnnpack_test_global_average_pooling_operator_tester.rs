// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/global-average-pooling-operator-tester.h]


pub struct GlobalAveragePoolingOperatorTester {
    batch_size:        usize, // default = { 1 }
    width:             usize, // default = { 1 }
    channels:          usize, // default = { 1 }
    input_stride:      usize, // default = { 0 }
    output_stride:     usize, // default = { 0 }
    input_scale:       f32, // default = { 1.0f }
    output_scale:      f32, // default = { 1.0f }
    input_zero_point:  u8, // default = { 121 }
    output_zero_point: u8, // default = { 133 }
    output_min:        u8, // default = { 0 }
    output_max:        u8, // default = { 255 }
    iterations:        usize, // default = { 1 }
}

impl GlobalAveragePoolingOperatorTester {

    
    #[inline] pub fn channels(&mut self, channels: usize) -> &mut GlobalAveragePoolingOperatorTester {
        
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
    
    #[inline] pub fn width(&mut self, width: usize) -> &mut GlobalAveragePoolingOperatorTester {
        
        todo!();
        /*
            assert(width != 0);
        this->width_ = width;
        return *this;
        */
    }
    
    #[inline] pub fn width(&self) -> usize {
        
        todo!();
        /*
            return this->width_;
        */
    }
    
    #[inline] pub fn input_stride(&mut self, input_stride: usize) -> &mut GlobalAveragePoolingOperatorTester {
        
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
          return channels();
        } else {
          assert(this->inputStride_ >= channels());
          return this->inputStride_;
        }
        */
    }
    
    #[inline] pub fn output_stride(&mut self, output_stride: usize) -> &mut GlobalAveragePoolingOperatorTester {
        
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
          return channels();
        } else {
          assert(this->outputStride_ >= channels());
          return this->outputStride_;
        }
        */
    }
    
    #[inline] pub fn batch_size(&mut self, batch_size: usize) -> &mut GlobalAveragePoolingOperatorTester {
        
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
    
    #[inline] pub fn input_scale(&mut self, input_scale: f32) -> &mut GlobalAveragePoolingOperatorTester {
        
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
    
    #[inline] pub fn input_zero_point(&mut self, input_zero_point: u8) -> &mut GlobalAveragePoolingOperatorTester {
        
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
    
    #[inline] pub fn output_scale(&mut self, output_scale: f32) -> &mut GlobalAveragePoolingOperatorTester {
        
        todo!();
        /*
            assert(outputScale > 0.0f);
        assert(isnormal(outputScale));
        this->outputScale_ = outputScale;
        return *this;
        */
    }
    
    #[inline] pub fn output_scale(&self) -> f32 {
        
        todo!();
        /*
            return this->outputScale_;
        */
    }
    
    #[inline] pub fn output_zero_point(&mut self, output_zero_point: u8) -> &mut GlobalAveragePoolingOperatorTester {
        
        todo!();
        /*
            this->outputZeroPoint_ = outputZeroPoint;
        return *this;
        */
    }
    
    #[inline] pub fn output_zero_point(&self) -> u8 {
        
        todo!();
        /*
            return this->outputZeroPoint_;
        */
    }
    
    #[inline] pub fn output_min(&mut self, output_min: u8) -> &mut GlobalAveragePoolingOperatorTester {
        
        todo!();
        /*
            this->outputMin_ = outputMin;
        return *this;
        */
    }
    
    #[inline] pub fn output_min(&self) -> u8 {
        
        todo!();
        /*
            return this->outputMin_;
        */
    }
    
    #[inline] pub fn output_max(&mut self, output_max: u8) -> &mut GlobalAveragePoolingOperatorTester {
        
        todo!();
        /*
            this->outputMax_ = outputMax;
        return *this;
        */
    }
    
    #[inline] pub fn output_max(&self) -> u8 {
        
        todo!();
        /*
            return this->outputMax_;
        */
    }
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut GlobalAveragePoolingOperatorTester {
        
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

        vector<u8> input(
            (batchSize() * width() - 1) * inputStride() + channels());
        vector<u8> output(batchSize() * outputStride());
        vector<float> outputRef(batchSize() * channels());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(input.begin(), input.end(), ref(u8rng));
          fill(output.begin(), output.end(), 0xA5);

          /* Compute reference results */
          const double scale =
              double(inputScale()) / (double(width()) * double(outputScale()));
          for (usize i = 0; i < batchSize(); i++) {
            for (usize j = 0; j < channels(); j++) {
              double acc = 0.0f;
              for (usize k = 0; k < width(); k++) {
                acc += double(
                    i32(input[(i * width() + k) * inputStride() + j]) -
                    i32(inputZeroPoint()));
              }
              outputRef[i * channels() + j] =
                  float(acc * scale + double(outputZeroPoint()));
              outputRef[i * channels() + j] = min<float>(
                  outputRef[i * channels() + j], float(outputMax()));
              outputRef[i * channels() + j] = max<float>(
                  outputRef[i * channels() + j], float(outputMin()));
            }
          }

          /* Create, setup, run, and destroy Add operator */
          ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
          pytorch_qnnp_operator_t globalAveragePoolingOp = nullptr;

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_create_global_average_pooling_nwc_q8(
                  channels(),
                  inputZeroPoint(),
                  inputScale(),
                  outputZeroPoint(),
                  outputScale(),
                  outputMin(),
                  outputMax(),
                  0,
                  &globalAveragePoolingOp));
          ASSERT_NE(nullptr, globalAveragePoolingOp);

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_setup_global_average_pooling_nwc_q8(
                  globalAveragePoolingOp,
                  batchSize(),
                  width(),
                  input.data(),
                  inputStride(),
                  output.data(),
                  outputStride()));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_run_operator(
                  globalAveragePoolingOp, nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_delete_operator(globalAveragePoolingOp));
          globalAveragePoolingOp = nullptr;

          /* Verify results */
          for (usize i = 0; i < batchSize(); i++) {
            for (usize c = 0; c < channels(); c++) {
              ASSERT_LE(
                  u32(output[i * outputStride() + c]), u32(outputMax()));
              ASSERT_GE(
                  u32(output[i * outputStride() + c]), u32(outputMin()));
              ASSERT_NEAR(
                  float(i32(output[i * outputStride() + c])),
                  outputRef[i * channels() + c],
                  0.80f)
                  << "in batch index " << i << ", channel " << c;
            }
          }
        }
        */
    }
}
