crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-operator-tester.h]

pub enum Mode {
    Static,
    Dynamic,
    Runtime,
}

pub struct FullyConnectedOperatorTester {
    input_channels:  usize, // default = { 1 }
    input_stride:    usize, // default = { 0 }
    output_channels: usize, // default = { 1 }
    output_stride:   usize, // default = { 0 }
    batch_size:      usize, // default = { 1 }
    qmin:            u8,    // default = { 0 }
    qmax:            u8,    // default = { 255 }
    iterations:      usize, // default = { 1 }
    per_channel:     bool,  // default = { false }
}

impl FullyConnectedOperatorTester {
    
    #[inline] pub fn input_channels(&mut self, input_channels: usize) -> &mut FullyConnectedOperatorTester {
        
        todo!();
        /*
            assert(inputChannels >= 1);
        this->inputChannels_ = inputChannels;
        return *this;
        */
    }
    
    #[inline] pub fn input_channels(&self) -> usize {
        
        todo!();
        /*
            return this->inputChannels_;
        */
    }
    
    #[inline] pub fn output_channels(&mut self, output_channels: usize) -> &mut FullyConnectedOperatorTester {
        
        todo!();
        /*
            assert(outputChannels >= 1);
        this->outputChannels_ = outputChannels;
        return *this;
        */
    }
    
    #[inline] pub fn output_channels(&self) -> usize {
        
        todo!();
        /*
            return this->outputChannels_;
        */
    }
    
    #[inline] pub fn batch_size(&mut self, batch_size: usize) -> &mut FullyConnectedOperatorTester {
        
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
    
    #[inline] pub fn input_stride(&mut self, input_stride: usize) -> &mut FullyConnectedOperatorTester {
        
        todo!();
        /*
            assert(inputStride >= 1);
        this->inputStride_ = inputStride;
        return *this;
        */
    }
    
    #[inline] pub fn input_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->inputStride_ == 0) {
          return inputChannels();
        } else {
          assert(this->inputStride_ >= inputChannels());
          return this->inputStride_;
        }
        */
    }
    
    #[inline] pub fn output_stride(&mut self, output_stride: usize) -> &mut FullyConnectedOperatorTester {
        
        todo!();
        /*
            assert(outputStride >= 1);
        this->outputStride_ = outputStride;
        return *this;
        */
    }
    
    #[inline] pub fn output_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->outputStride_ == 0) {
          return outputChannels();
        } else {
          assert(this->outputStride_ >= outputChannels());
          return this->outputStride_;
        }
        */
    }
    
    #[inline] pub fn per_channel(&mut self, per_channel: bool) -> &mut FullyConnectedOperatorTester {
        
        todo!();
        /*
            this->per_channel_ = per_channel;
        return *this;
        */
    }
    
    #[inline] pub fn per_channel(&self) -> bool {
        
        todo!();
        /*
            return this->per_channel_;
        */
    }
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut FullyConnectedOperatorTester {
        
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
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut FullyConnectedOperatorTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut FullyConnectedOperatorTester {
        
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
    
    pub fn testq8(&self, mode: Mode)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);
        auto f32rng =
            bind(uniform_real_distribution<float>(1, 5), rng);

        vector<u8> input(
            (batchSize() - 1) * inputStride() + inputChannels() + 8);
        vector<u8> kernel(outputChannels() * inputChannels());
        vector<i32> bias(outputChannels());
        vector<u8> output(
            (batchSize() - 1) * outputStride() + outputChannels());
        vector<float> output_dynamic(output.size());
        vector<i32> accumulators(batchSize() * outputChannels());

        const u8* const inputPtr = input.data() + 8;
        const u8 inputZeroPoint = 127;
        // Make number of output channels multiple of 8.
        // This is the least common denominator for SSE/ARM kernels we have.
        usize num_zero_points_padded = outputChannels() + 8;
        vector<u8> kernelZeroPoints(num_zero_points_padded, 127);

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(input.begin(), input.end(), ref(u8rng));
          generate(kernel.begin(), kernel.end(), ref(u8rng));
          generate(bias.begin(), bias.end(), ref(s32rng));
          if (per_channel()) {
            generate(kernelZeroPoints.begin(), kernelZeroPoints.end(), ref(u8rng));
          }
          fill(output.begin(), output.end(), 0xA5);
          fill(output_dynamic.begin(), output_dynamic.end(), 0.0f);
          fill(accumulators.begin(), accumulators.end(), 0);

          for (usize i = 0; i < batchSize(); i++) {
            for (usize oc = 0; oc < outputChannels(); oc++) {
              accumulators[i * outputChannels() + oc] = bias[oc];
            }
          }
          for (usize i = 0; i < batchSize(); i++) {
            for (usize oc = 0; oc < outputChannels(); oc++) {
              for (usize ic = 0; ic < inputChannels(); ic++) {
                accumulators[i * outputChannels() + oc] +=
                    (i32(inputPtr[i * inputStride() + ic]) -
                     i32(inputZeroPoint)) *
                    (i32(kernel[oc * inputChannels() + ic]) -
                     i32(kernelZeroPoints[oc]));
              }
            }
          }

          // Create dummy min/max for empty inputs.
          // These are only used to compute scale and zero point,
          // and real callers will just pull those values from the model.
          const i32 accumulatorsMin = accumulators.empty()
              ? 0
              : *min_element(accumulators.cbegin(), accumulators.cend());
          const i32 accumulatorsMax = accumulators.empty()
              ? 900
              : *max_element(accumulators.cbegin(), accumulators.cend());

          const double outputScale =
              double(u32(accumulatorsMax - accumulatorsMin)) / 255.0;
          const u8 outputZeroPoint = u8(max(
              min(
                  lrint(
                      127.5 -
                      0.5 * double(accumulatorsMin + accumulatorsMax) /
                          outputScale),
                  long(u8::max)),
              long(u8::min)));

          ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
          // 1 bcz input_scale and kernel_scale are both 1.
          vector<float>
            requantization_scales(num_zero_points_padded, 1.0 * 1.0 / outputScale);
          if (per_channel()) {
            auto scale_generator = [&]() -> float {return (f32rng()/outputScale);};
            generate(
                requantization_scales.begin(),
                requantization_scales.end(),
                ref(scale_generator));
          }

          switch(mode) {
            case Mode::Static:
            {
              pytorch_qnnp_operator_t convolution = nullptr;

              ASSERT_EQ(
                  pytorch_qnnp_status_success,
                  pytorch_qnnp_create_fully_connected_nc_q8(
                      inputChannels(),
                      outputChannels(),
                      inputZeroPoint,
                      kernelZeroPoints.data(),
                      kernel.data(),
                      bias.data(),
                      outputZeroPoint,
                      qmin(),
                      qmax(),
                      0,
                      requantization_scales.data(),
                      &convolution));

              ASSERT_EQ(
                  pytorch_qnnp_status_success,
                  pytorch_qnnp_setup_fully_connected_nc_q8(
                      convolution,
                      batchSize(),
                      inputPtr,
                      inputStride(),
                      output.data(),
                      outputStride()));

              ASSERT_EQ(
                  pytorch_qnnp_status_success,
                  pytorch_qnnp_run_operator(convolution, nullptr /* thread pool */));

              ASSERT_EQ(
                  pytorch_qnnp_status_success,
                  pytorch_qnnp_delete_operator(convolution));
              convolution = nullptr;
            }
            break;

            case Mode::Dynamic:
            {
              auto packW = unique_ptr<qnnpack_PackBMatrix>(
                  new qnnpack_PackBMatrix(
                      inputChannels(),
                      outputChannels(),
                      kernelZeroPoints.data(),
                      requantization_scales.data(),
                      kernel.data(),
                      nullptr));

              // Attention! Bias size must be a multiple of 8.
              constexpr usize kBiasSizeMultiple = 8u;
              vector<float, AlignedAllocator<float, 32>> bias_float(
                (bias.size() + (kBiasSizeMultiple - 1)) & -kBiasSizeMultiple);
              copy(bias.cbegin(), bias.cend(), bias_float.begin());

              const pytorch_qnnp_status runStatus = qnnpack_qnnpackLinearDynamic(
                  batchSize() /* batch_size */,
                  inputChannels() /* input_channels */,
                  outputChannels() /* output_channels */,
                  inputZeroPoint,
                  kernelZeroPoints.data(),
                  requantization_scales.data(), /* Dequantization scale */
                  inputPtr,
                  inputChannels() /* input_stride */,
                  packW->getPackedWeights(),
                  bias_float.data(),
                  output_dynamic.data(),
                  outputStride() /* output_stride */,
                  nullptr /* threadpool */);
              ASSERT_EQ(pytorch_qnnp_status_success, runStatus);
            }
            break;

            case Mode::Runtime:
            {
              auto packW = unique_ptr<qnnpack_PackBMatrix>(
                  new qnnpack_PackBMatrix(
                      inputChannels(),
                      outputChannels(),
                      kernelZeroPoints.data(),
                      requantization_scales.data(),
                      kernel.data(),
                      bias.data()));

              const pytorch_qnnp_status runStatus = qnnpack_qnnpackLinear(
                  batchSize() /* batch_size */,
                  inputChannels() /* input_channels */,
                  outputChannels() /* output_channels */,
                  inputZeroPoint,
                  kernelZeroPoints.data(),
                  requantization_scales.data(),
                  outputZeroPoint,
                  qmin(),
                  qmax(),
                  inputPtr,
                  inputChannels() /* input_stride */,
                  packW->getPackedWeights(),
                  output.data(),
                  outputStride() /* output_stride */,
                  nullptr /* threadpool */);
              ASSERT_EQ(pytorch_qnnp_status_success, runStatus);
            }
            break;

            default:
              // Undefined!
              ASSERT_TRUE(false);
          }

          switch (mode) {
            case Mode::Static:
            case Mode::Runtime:
            {
              for (usize i = 0; i < batchSize(); i++) {
                for (usize c = 0; c < outputChannels(); c++) {
                  const double scaledAccumulator =
                      accumulators[i * outputChannels() + c] *
                      requantization_scales[c];
                  const double clampedAccumulator = max(
                      min(
                          scaledAccumulator, double(qmax()) - double(outputZeroPoint)),
                      double(qmin()) - double(outputZeroPoint));
                  ASSERT_NEAR(
                      clampedAccumulator,
                      (i32(output[i * outputStride() + c]) - outputZeroPoint),
                      0.9)
                      << "batch index = " << i << ", channel = " << c;
                }
              }
            }
            break;

            case Mode::Dynamic:
            {
              // Bias is added post scaling, as float.
              for (usize i = 0; i < batchSize(); i++) {
                for (usize oc = 0; oc < outputChannels(); oc++) {
                  accumulators[i * outputChannels() + oc] -= bias[oc];
                }
              }
              for (usize i = 0; i < batchSize(); i++) {
                for (usize c = 0; c < outputChannels(); c++) {
                  ASSERT_EQ(
                      output_dynamic[i * outputChannels() + c],
                      ((float)accumulators[i * outputChannels() + c] *
                      requantization_scales[c]) + float(bias[c]))
                      << "at " << i << ", " << c
                      << ": reference = " <<
                      ((float)accumulators[i * outputChannels() + c] *
                      requantization_scales[c]) + float(bias[c])
                      << ", optimized = " << output_dynamic[i * outputChannels() + c];
                }
              }
            }
            break;

            default:
              // Undefined!
              ASSERT_TRUE(false);
          }
        }
        */
    }
}
