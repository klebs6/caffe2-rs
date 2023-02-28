// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/max-pooling-operator-tester.h]

pub struct MaxPoolingOperatorTester {
    padding_top:         u32, // default = { 0 }
    padding_right:       u32, // default = { 0 }
    padding_bottom:      u32, // default = { 0 }
    padding_left:        u32, // default = { 0 }
    input_height:        usize, // default = { 1 }
    input_width:         usize, // default = { 1 }
    channels:            usize, // default = { 1 }
    batch_size:          usize, // default = { 1 }
    input_pixel_stride:  usize, // default = { 0 }
    output_pixel_stride: usize, // default = { 0 }
    pooling_height:      u32, // default = { 1 }
    pooling_width:       u32, // default = { 1 }
    stride_height:       u32, // default = { 1 }
    stride_width:        u32, // default = { 1 }
    dilation_height:     u32, // default = { 1 }
    dilation_width:      u32, // default = { 1 }
    next_input_height:   usize, // default = { 0 }
    next_input_width:    usize, // default = { 0 }
    next_batch_size:     usize, // default = { 0 }
    qmin:                u8, // default = { 0 }
    qmax:                u8, // default = { 255 }
    iterations:          usize, // default = { 1 }
}

impl MaxPoolingOperatorTester {

    
    #[inline] pub fn padding(&mut self, padding: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            this->paddingTop_ = padding;
        this->paddingRight_ = padding;
        this->paddingBottom_ = padding;
        this->paddingLeft_ = padding;
        return *this;
        */
    }
    
    #[inline] pub fn padding(&mut self, 
        padding_height: u32,
        padding_width:  u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            this->paddingTop_ = paddingHeight;
        this->paddingRight_ = paddingWidth;
        this->paddingBottom_ = paddingHeight;
        this->paddingLeft_ = paddingWidth;
        return *this;
        */
    }
    
    #[inline] pub fn padding_height(&mut self, padding_height: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            this->paddingTop_ = paddingHeight;
        this->paddingBottom_ = paddingHeight;
        return *this;
        */
    }
    
    #[inline] pub fn padding_width(&mut self, padding_width: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            this->paddingRight_ = paddingWidth;
        this->paddingLeft_ = paddingWidth;
        return *this;
        */
    }
    
    #[inline] pub fn padding_top(&mut self, padding_top: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            this->paddingTop_ = paddingTop;
        return *this;
        */
    }
    
    #[inline] pub fn padding_top(&self) -> u32 {
        
        todo!();
        /*
            return this->paddingTop_;
        */
    }
    
    #[inline] pub fn padding_right(&mut self, padding_right: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            this->paddingRight_ = paddingRight;
        return *this;
        */
    }
    
    #[inline] pub fn padding_right(&self) -> u32 {
        
        todo!();
        /*
            return this->paddingRight_;
        */
    }
    
    #[inline] pub fn padding_bottom(&mut self, padding_bottom: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            this->paddingBottom_ = paddingBottom;
        return *this;
        */
    }
    
    #[inline] pub fn padding_bottom(&self) -> u32 {
        
        todo!();
        /*
            return this->paddingBottom_;
        */
    }
    
    #[inline] pub fn padding_left(&mut self, padding_left: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            this->paddingLeft_ = paddingLeft;
        return *this;
        */
    }
    
    #[inline] pub fn padding_left(&self) -> u32 {
        
        todo!();
        /*
            return this->paddingLeft_;
        */
    }
    
    #[inline] pub fn input_size(&mut self, 
        input_height: usize,
        input_width:  usize) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(inputHeight >= 1);
        assert(inputWidth >= 1);
        this->inputHeight_ = inputHeight;
        this->inputWidth_ = inputWidth;
        return *this;
        */
    }
    
    #[inline] pub fn input_height(&mut self, input_height: usize) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(inputHeight >= 1);
        this->inputHeight_ = inputHeight;
        return *this;
        */
    }
    
    #[inline] pub fn input_height(&self) -> usize {
        
        todo!();
        /*
            return this->inputHeight_;
        */
    }
    
    #[inline] pub fn input_width(&mut self, input_width: usize) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(inputWidth >= 1);
        this->inputWidth_ = inputWidth;
        return *this;
        */
    }
    
    #[inline] pub fn input_width(&self) -> usize {
        
        todo!();
        /*
            return this->inputWidth_;
        */
    }
    
    #[inline] pub fn channels(&mut self, channels: usize) -> &mut MaxPoolingOperatorTester {
        
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
    
    #[inline] pub fn batch_size(&mut self, batch_size: usize) -> &mut MaxPoolingOperatorTester {
        
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
    
    #[inline] pub fn pooling_size(&mut self, pooling_size: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(poolingSize >= 1);
        this->poolingHeight_ = poolingSize;
        this->poolingWidth_ = poolingSize;
        return *this;
        */
    }
    
    #[inline] pub fn pooling_size(&mut self, 
        pooling_height: u32,
        pooling_width:  u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(poolingHeight >= 1);
        assert(poolingWidth >= 1);
        this->poolingHeight_ = poolingHeight;
        this->poolingWidth_ = poolingWidth;
        return *this;
        */
    }
    
    #[inline] pub fn pooling_height(&mut self, pooling_height: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(poolingHeight >= 1);
        this->poolingHeight_ = poolingHeight;
        return *this;
        */
    }
    
    #[inline] pub fn pooling_height(&self) -> u32 {
        
        todo!();
        /*
            return this->poolingHeight_;
        */
    }
    
    #[inline] pub fn pooling_width(&mut self, pooling_width: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(poolingWidth >= 1);
        this->poolingWidth_ = poolingWidth;
        return *this;
        */
    }
    
    #[inline] pub fn pooling_width(&self) -> u32 {
        
        todo!();
        /*
            return this->poolingWidth_;
        */
    }
    
    #[inline] pub fn stride(&mut self, stride: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(stride >= 1);
        this->strideHeight_ = stride;
        this->strideWidth_ = stride;
        return *this;
        */
    }
    
    #[inline] pub fn stride(&mut self, 
        stride_height: u32,
        stride_width:  u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(strideHeight >= 1);
        assert(strideWidth >= 1);
        this->strideHeight_ = strideHeight;
        this->strideWidth_ = strideWidth;
        return *this;
        */
    }
    
    #[inline] pub fn stride_height(&mut self, stride_height: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(strideHeight >= 1);
        this->strideHeight_ = strideHeight;
        return *this;
        */
    }
    
    #[inline] pub fn stride_height(&self) -> u32 {
        
        todo!();
        /*
            return this->strideHeight_;
        */
    }
    
    #[inline] pub fn stride_width(&mut self, stride_width: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(strideWidth >= 1);
        this->strideWidth_ = strideWidth;
        return *this;
        */
    }
    
    #[inline] pub fn stride_width(&self) -> u32 {
        
        todo!();
        /*
            return this->strideWidth_;
        */
    }
    
    #[inline] pub fn dilation(&mut self, dilation: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(dilation >= 1);
        this->dilationHeight_ = dilation;
        this->dilationWidth_ = dilation;
        return *this;
        */
    }
    
    #[inline] pub fn dilation(&mut self, 
        dilation_height: u32,
        dilation_width:  u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(dilationHeight >= 1);
        assert(dilationWidth >= 1);
        this->dilationHeight_ = dilationHeight;
        this->dilationWidth_ = dilationWidth;
        return *this;
        */
    }
    
    #[inline] pub fn dilation_height(&mut self, dilation_height: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(dilationHeight >= 1);
        this->dilationHeight_ = dilationHeight;
        return *this;
        */
    }
    
    #[inline] pub fn dilation_height(&self) -> u32 {
        
        todo!();
        /*
            return this->dilationHeight_;
        */
    }
    
    #[inline] pub fn dilation_width(&mut self, dilation_width: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(dilationWidth >= 1);
        this->dilationWidth_ = dilationWidth;
        return *this;
        */
    }
    
    #[inline] pub fn dilation_width(&self) -> u32 {
        
        todo!();
        /*
            return this->dilationWidth_;
        */
    }
    
    #[inline] pub fn dilated_pooling_height(&self) -> u32 {
        
        todo!();
        /*
            return (poolingHeight() - 1) * dilationHeight() + 1;
        */
    }
    
    #[inline] pub fn dilated_pooling_width(&self) -> u32 {
        
        todo!();
        /*
            return (poolingWidth() - 1) * dilationWidth() + 1;
        */
    }
    
    #[inline] pub fn output_height(&self) -> usize {
        
        todo!();
        /*
            const usize paddedInputHeight =
            paddingTop() + inputHeight() + paddingBottom();
        if (paddedInputHeight <= dilatedPoolingHeight()) {
          return 1;
        } else {
          return (paddedInputHeight - dilatedPoolingHeight()) / strideHeight() + 1;
        }
        */
    }
    
    #[inline] pub fn output_width(&self) -> usize {
        
        todo!();
        /*
            const usize paddedInputWidth =
            paddingLeft() + inputWidth() + paddingRight();
        if (paddedInputWidth <= dilatedPoolingWidth()) {
          return 1;
        } else {
          return (paddedInputWidth - dilatedPoolingWidth()) / strideWidth() + 1;
        }
        */
    }
    
    #[inline] pub fn input_pixel_stride(&mut self, input_pixel_stride: usize) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(inputPixelStride != 0);
        this->inputPixelStride_ = inputPixelStride;
        return *this;
        */
    }
    
    #[inline] pub fn input_pixel_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->inputPixelStride_ == 0) {
          return channels();
        } else {
          assert(this->inputPixelStride_ >= channels());
          return this->inputPixelStride_;
        }
        */
    }
    
    #[inline] pub fn output_pixel_stride(&mut self, output_pixel_stride: usize) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(outputPixelStride != 0);
        this->outputPixelStride_ = outputPixelStride;
        return *this;
        */
    }
    
    #[inline] pub fn output_pixel_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->outputPixelStride_ == 0) {
          return channels();
        } else {
          assert(this->outputPixelStride_ >= channels());
          return this->outputPixelStride_;
        }
        */
    }
    
    #[inline] pub fn next_input_size(&mut self, 
        next_input_height: u32,
        next_input_width:  u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(nextInputHeight >= 1);
        assert(nextInputWidth >= 1);
        this->nextInputHeight_ = nextInputHeight;
        this->nextInputWidth_ = nextInputWidth;
        return *this;
        */
    }
    
    #[inline] pub fn next_input_height(&mut self, next_input_height: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(nextInputHeight >= 1);
        this->nextInputHeight_ = nextInputHeight;
        return *this;
        */
    }
    
    #[inline] pub fn next_input_height(&self) -> u32 {
        
        todo!();
        /*
            if (this->nextInputHeight_ == 0) {
          return inputHeight();
        } else {
          return this->nextInputHeight_;
        }
        */
    }
    
    #[inline] pub fn next_input_width(&mut self, next_input_width: u32) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(nextInputWidth >= 1);
        this->nextInputWidth_ = nextInputWidth;
        return *this;
        */
    }
    
    #[inline] pub fn next_input_width(&self) -> u32 {
        
        todo!();
        /*
            if (this->nextInputWidth_ == 0) {
          return inputWidth();
        } else {
          return this->nextInputWidth_;
        }
        */
    }
    
    #[inline] pub fn next_output_height(&self) -> usize {
        
        todo!();
        /*
            const usize paddedNextInputHeight =
            paddingTop() + nextInputHeight() + paddingBottom();
        if (paddedNextInputHeight <= dilatedPoolingHeight()) {
          return 1;
        } else {
          return (paddedNextInputHeight - dilatedPoolingHeight()) / strideHeight() +
              1;
        }
        */
    }
    
    #[inline] pub fn next_output_width(&self) -> usize {
        
        todo!();
        /*
            const usize paddedNextInputWidth =
            paddingLeft() + nextInputWidth() + paddingRight();
        if (paddedNextInputWidth <= dilatedPoolingWidth()) {
          return 1;
        } else {
          return (paddedNextInputWidth - dilatedPoolingWidth()) / strideWidth() + 1;
        }
        */
    }
    
    #[inline] pub fn next_batch_size(&mut self, next_batch_size: usize) -> &mut MaxPoolingOperatorTester {
        
        todo!();
        /*
            assert(nextBatchSize >= 1);
        this->nextBatchSize_ = nextBatchSize;
        return *this;
        */
    }
    
    #[inline] pub fn next_batch_size(&self) -> usize {
        
        todo!();
        /*
            if (this->nextBatchSize_ == 0) {
          return batchSize();
        } else {
          return this->nextBatchSize_;
        }
        */
    }
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut MaxPoolingOperatorTester {
        
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
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut MaxPoolingOperatorTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut MaxPoolingOperatorTester {
        
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

        vector<u8> input(
            (batchSize() * inputHeight() * inputWidth() - 1) * inputPixelStride() +
            channels());
        vector<u8> output(
            (batchSize() * outputHeight() * outputWidth() - 1) *
                outputPixelStride() +
            channels());
        vector<u8> outputRef(
            batchSize() * outputHeight() * outputWidth() * channels());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(input.begin(), input.end(), ref(u8rng));
          fill(output.begin(), output.end(), 0xA5);

          /* Compute reference results */
          for (usize i = 0; i < batchSize(); i++) {
            for (usize oy = 0; oy < outputHeight(); oy++) {
              for (usize ox = 0; ox < outputWidth(); ox++) {
                for (usize c = 0; c < channels(); c++) {
                  u8 maxValue = 0;
                  for (usize py = 0; py < poolingHeight(); py++) {
                    const usize iy =
                        oy * strideHeight() + py * dilationHeight() - paddingTop();
                    for (usize px = 0; px < poolingWidth(); px++) {
                      const usize ix =
                          ox * strideWidth() + px * dilationWidth() - paddingLeft();
                      if (ix < inputWidth() && iy < inputHeight()) {
                        maxValue = max(
                            maxValue,
                            input
                                [((i * inputHeight() + iy) * inputWidth() + ix) *
                                     inputPixelStride() +
                                 c]);
                      }
                    }
                  }
                  maxValue = min(maxValue, qmax());
                  maxValue = max(maxValue, qmin());
                  outputRef
                      [((i * outputHeight() + oy) * outputWidth() + ox) *
                           channels() +
                       c] = maxValue;
                }
              }
            }
          }

          /* Create, setup, run, and destroy Max Pooling operator */
          ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
          pytorch_qnnp_operator_t maxPoolingOp = nullptr;

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_create_max_pooling2d_nhwc_u8(
                  paddingTop(),
                  paddingRight(),
                  paddingBottom(),
                  paddingLeft(),
                  poolingHeight(),
                  poolingWidth(),
                  strideHeight(),
                  strideWidth(),
                  dilationHeight(),
                  dilationWidth(),
                  channels(),
                  qmin(),
                  qmax(),
                  0,
                  &maxPoolingOp));
          ASSERT_NE(nullptr, maxPoolingOp);

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
                  maxPoolingOp,
                  batchSize(),
                  inputHeight(),
                  inputWidth(),
                  input.data(),
                  inputPixelStride(),
                  output.data(),
                  outputPixelStride(),
                  nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_run_operator(maxPoolingOp, nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_delete_operator(maxPoolingOp));
          maxPoolingOp = nullptr;

          /* Verify results */
          for (usize i = 0; i < batchSize(); i++) {
            for (usize y = 0; y < outputHeight(); y++) {
              for (usize x = 0; x < outputWidth(); x++) {
                for (usize c = 0; c < channels(); c++) {
                  ASSERT_LE(
                      u32(output
                                   [((i * outputHeight() + y) * outputWidth() + x) *
                                        outputPixelStride() +
                                    c]),
                      u32(qmax()));
                  ASSERT_GE(
                      u32(output
                                   [((i * outputHeight() + y) * outputWidth() + x) *
                                        outputPixelStride() +
                                    c]),
                      u32(qmin()));
                  ASSERT_EQ(
                      u32(outputRef
                                   [((i * outputHeight() + y) * outputWidth() + x) *
                                        channels() +
                                    c]),
                      u32(output
                                   [((i * outputHeight() + y) * outputWidth() + x) *
                                        outputPixelStride() +
                                    c]))
                      << "in batch index " << i << ", pixel (" << y << ", " << x
                      << "), channel " << c;
                }
              }
            }
          }
        }
        */
    }
    
    pub fn test_setupu8(&self)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> input(max(
            (batchSize() * inputHeight() * inputWidth() - 1) * inputPixelStride() +
                channels(),
            (nextBatchSize() * nextInputHeight() * nextInputWidth() - 1) *
                    inputPixelStride() +
                channels()));
        vector<u8> output(max(
            (batchSize() * outputHeight() * outputWidth() - 1) *
                    outputPixelStride() +
                channels(),
            (nextBatchSize() * nextOutputHeight() * nextOutputWidth() - 1) *
                    outputPixelStride() +
                channels()));
        vector<float> outputRef(
            batchSize() * outputHeight() * outputWidth() * channels());
        vector<float> nextOutputRef(
            nextBatchSize() * nextOutputHeight() * nextOutputWidth() * channels());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(input.begin(), input.end(), ref(u8rng));
          fill(output.begin(), output.end(), 0xA5);

          /* Compute reference results */
          for (usize i = 0; i < batchSize(); i++) {
            for (usize oy = 0; oy < outputHeight(); oy++) {
              for (usize ox = 0; ox < outputWidth(); ox++) {
                for (usize c = 0; c < channels(); c++) {
                  u8 maxValue = 0;
                  for (usize py = 0; py < poolingHeight(); py++) {
                    const usize iy =
                        oy * strideHeight() + py * dilationHeight() - paddingTop();
                    for (usize px = 0; px < poolingWidth(); px++) {
                      const usize ix =
                          ox * strideWidth() + px * dilationWidth() - paddingLeft();
                      if (ix < inputWidth() && iy < inputHeight()) {
                        maxValue = max(
                            maxValue,
                            input
                                [((i * inputHeight() + iy) * inputWidth() + ix) *
                                     inputPixelStride() +
                                 c]);
                      }
                    }
                  }
                  maxValue = min(maxValue, qmax());
                  maxValue = max(maxValue, qmin());
                  outputRef
                      [((i * outputHeight() + oy) * outputWidth() + ox) *
                           channels() +
                       c] = maxValue;
                }
              }
            }
          }

          /* Create, setup, and run Max Pooling operator once */
          ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
          pytorch_qnnp_operator_t maxPoolingOp = nullptr;

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_create_max_pooling2d_nhwc_u8(
                  paddingTop(),
                  paddingRight(),
                  paddingBottom(),
                  paddingLeft(),
                  poolingHeight(),
                  poolingWidth(),
                  strideHeight(),
                  strideWidth(),
                  dilationHeight(),
                  dilationWidth(),
                  channels(),
                  qmin(),
                  qmax(),
                  0,
                  &maxPoolingOp));
          ASSERT_NE(nullptr, maxPoolingOp);

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
                  maxPoolingOp,
                  batchSize(),
                  inputHeight(),
                  inputWidth(),
                  input.data(),
                  inputPixelStride(),
                  output.data(),
                  outputPixelStride(),
                  nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_run_operator(maxPoolingOp, nullptr /* thread pool */));

          /* Verify results of the first run */
          for (usize i = 0; i < batchSize(); i++) {
            for (usize y = 0; y < outputHeight(); y++) {
              for (usize x = 0; x < outputWidth(); x++) {
                for (usize c = 0; c < channels(); c++) {
                  ASSERT_LE(
                      u32(output
                                   [((i * outputHeight() + y) * outputWidth() + x) *
                                        outputPixelStride() +
                                    c]),
                      u32(qmax()));
                  ASSERT_GE(
                      u32(output
                                   [((i * outputHeight() + y) * outputWidth() + x) *
                                        outputPixelStride() +
                                    c]),
                      u32(qmin()));
                  ASSERT_EQ(
                      u32(outputRef
                                   [((i * outputHeight() + y) * outputWidth() + x) *
                                        channels() +
                                    c]),
                      u32(output
                                   [((i * outputHeight() + y) * outputWidth() + x) *
                                        outputPixelStride() +
                                    c]))
                      << "in batch index " << i << ", pixel (" << y << ", " << x
                      << "), channel " << c;
                }
              }
            }
          }

          /* Re-generate data for the second run */
          generate(input.begin(), input.end(), ref(u8rng));
          fill(output.begin(), output.end(), 0xA5);

          /* Compute reference results for the second run */
          for (usize i = 0; i < nextBatchSize(); i++) {
            for (usize oy = 0; oy < nextOutputHeight(); oy++) {
              for (usize ox = 0; ox < nextOutputWidth(); ox++) {
                for (usize c = 0; c < channels(); c++) {
                  u8 maxValue = 0;
                  for (usize py = 0; py < poolingHeight(); py++) {
                    const usize iy =
                        oy * strideHeight() + py * dilationHeight() - paddingTop();
                    for (usize px = 0; px < poolingWidth(); px++) {
                      const usize ix =
                          ox * strideWidth() + px * dilationWidth() - paddingLeft();
                      if (ix < nextInputWidth() && iy < nextInputHeight()) {
                        maxValue = max(
                            maxValue,
                            input
                                [((i * nextInputHeight() + iy) * nextInputWidth() +
                                  ix) *
                                     inputPixelStride() +
                                 c]);
                      }
                    }
                  }
                  maxValue = min(maxValue, qmax());
                  maxValue = max(maxValue, qmin());
                  nextOutputRef
                      [((i * nextOutputHeight() + oy) * nextOutputWidth() + ox) *
                           channels() +
                       c] = maxValue;
                }
              }
            }
          }

          /* Setup and run Max Pooling operator the second time, and destroy the
           * operator */
          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
                  maxPoolingOp,
                  nextBatchSize(),
                  nextInputHeight(),
                  nextInputWidth(),
                  input.data(),
                  inputPixelStride(),
                  output.data(),
                  outputPixelStride(),
                  nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_run_operator(maxPoolingOp, nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_delete_operator(maxPoolingOp));
          maxPoolingOp = nullptr;

          /* Verify results of the second run */
          for (usize i = 0; i < nextBatchSize(); i++) {
            for (usize y = 0; y < nextOutputHeight(); y++) {
              for (usize x = 0; x < nextOutputWidth(); x++) {
                for (usize c = 0; c < channels(); c++) {
                  ASSERT_LE(
                      u32(
                          output
                              [((i * nextOutputHeight() + y) * nextOutputWidth() +
                                x) *
                                   outputPixelStride() +
                               c]),
                      u32(qmax()));
                  ASSERT_GE(
                      u32(
                          output
                              [((i * nextOutputHeight() + y) * nextOutputWidth() +
                                x) *
                                   outputPixelStride() +
                               c]),
                      u32(qmin()));
                  ASSERT_EQ(
                      u32(
                          nextOutputRef
                              [((i * nextOutputHeight() + y) * nextOutputWidth() +
                                x) *
                                   channels() +
                               c]),
                      u32(
                          output
                              [((i * nextOutputHeight() + y) * nextOutputWidth() +
                                x) *
                                   outputPixelStride() +
                               c]))
                      << "in batch index " << i << ", pixel (" << y << ", " << x
                      << "), channel " << c;
                }
              }
            }
          }
        }
        */
    }
}
