// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/dwconv-microkernel-tester.h]

pub struct DWConvMicrokernelTester {
    channels:          u32, // default = { 1 }
    cr:                u32, // default = { 1 }
    width:             u32, // default = { 1 }
    subsampling:       u32, // default = { 1 }
    kernel_height:     u32, // default = { 1 }
    kernel_width:      u32, // default = { 1 }
    input_stride:      u32, // default = { 0 }
    output_stride:     u32, // default = { 0 }
    input_zero_point:  u8, // default = { 127 }
    kernel_zero_point: u8, // default = { 127 }
    qmin:              u8, // default = { 0 }
    qmax:              u8, // default = { 255 }
    iterations:        usize, // default = { 3 }
}

impl DWConvMicrokernelTester {
    
    #[inline] pub fn width(&mut self, width: u32) -> &mut DWConvMicrokernelTester {
        
        todo!();
        /*
            assert(width >= 1);
        this->width_ = width;
        return *this;
        */
    }
    
    #[inline] pub fn width(&self) -> u32 {
        
        todo!();
        /*
            return this->width_;
        */
    }
    
    #[inline] pub fn subsampling(&mut self, subsampling: u32) -> &mut DWConvMicrokernelTester {
        
        todo!();
        /*
            assert(subsampling >= 1);
        this->subsampling_ = subsampling;
        return *this;
        */
    }
    
    #[inline] pub fn subsampling(&self) -> u32 {
        
        todo!();
        /*
            return this->subsampling_;
        */
    }
    
    #[inline] pub fn channels(&mut self, channels: u32) -> &mut DWConvMicrokernelTester {
        
        todo!();
        /*
            assert(channels >= 1);
        this->channels_ = channels;
        return *this;
        */
    }
    
    #[inline] pub fn channels(&self) -> u32 {
        
        todo!();
        /*
            return this->channels_;
        */
    }
    
    #[inline] pub fn cr(&mut self, cr: u32) -> &mut DWConvMicrokernelTester {
        
        todo!();
        /*
            assert(cr != 0);
        assert((cr & (cr - 1)) == 0);
        this->cr_ = cr;
        return *this;
        */
    }
    
    #[inline] pub fn cr(&self) -> u32 {
        
        todo!();
        /*
            return this->cr_;
        */
    }
    
    #[inline] pub fn packed_channels(&self) -> u32 {
        
        todo!();
        /*
            return (channels() + (cr() - 1)) & -cr();
        */
    }
    
    #[inline] pub fn kernel_height(&mut self, kernel_height: u32) -> &mut DWConvMicrokernelTester {
        
        todo!();
        /*
            assert(kernelHeight != 0);
        this->kernelHeight_ = kernelHeight;
        return *this;
        */
    }
    
    #[inline] pub fn kernel_height(&self) -> u32 {
        
        todo!();
        /*
            return this->kernelHeight_;
        */
    }
    
    #[inline] pub fn kernel_width(&mut self, kernel_width: u32) -> &mut DWConvMicrokernelTester {
        
        todo!();
        /*
            assert(kernelWidth != 0);
        this->kernelWidth_ = kernelWidth;
        return *this;
        */
    }
    
    #[inline] pub fn kernel_width(&self) -> u32 {
        
        todo!();
        /*
            return this->kernelWidth_;
        */
    }
    
    #[inline] pub fn kernel_size(&self) -> u32 {
        
        todo!();
        /*
            return kernelHeight() * kernelWidth();
        */
    }
    
    #[inline] pub fn input_stride(&mut self, input_stride: u32) -> &mut DWConvMicrokernelTester {
        
        todo!();
        /*
            assert(inputStride != 0);
        this->inputStride_ = inputStride;
        return *this;
        */
    }
    
    #[inline] pub fn input_stride(&self) -> u32 {
        
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
    
    #[inline] pub fn output_stride(&mut self, output_stride: u32) -> &mut DWConvMicrokernelTester {
        
        todo!();
        /*
            assert(outputStride != 0);
        this->outputStride_ = outputStride;
        return *this;
        */
    }
    
    #[inline] pub fn output_stride(&self) -> u32 {
        
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
    
    #[inline] pub fn input_zero_point(&mut self, input_zero_point: u8) -> &mut DWConvMicrokernelTester {
        
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
    
    #[inline] pub fn kernel_zero_point(&mut self, kernel_zero_point: u8) -> &mut DWConvMicrokernelTester {
        
        todo!();
        /*
            this->kernelZeroPoint_ = kernelZeroPoint;
        return *this;
        */
    }
    
    #[inline] pub fn kernel_zero_point(&self) -> u8 {
        
        todo!();
        /*
            return this->kernelZeroPoint_;
        */
    }
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut DWConvMicrokernelTester {
        
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
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut DWConvMicrokernelTester {
        
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
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut DWConvMicrokernelTester {
        
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
    
    pub fn test(&self, 
        q8dwconv:    PyTorchQ8DwConvUpUKernelFunction,
        per_channel: bool)  {
        let per_channel: bool = per_channel.unwrap_or(false);

        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> input(
            (kernelSize() + (width() * subsampling() - 1) * kernelHeight() - 1) *
                inputStride() +
            channels() + 8);
        vector<u8> kernel(channels() * kernelSize());
        vector<u8, AlignedAllocator<u8, 32>> packedWeights(
            (kernelSize() + sizeof(i32) / sizeof(u8)) * packedChannels());
        vector<i32> bias(packedChannels());
        vector<i32> accumulators(width() * channels());
        vector<u8> output((width() - 1) * outputStride() + channels());
        vector<const u8*> indirectInput(
            kernelSize() + (width() * subsampling() - 1) * kernelHeight());

        const u8* inputPtr = input.data() + 8;

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(input.begin(), input.end(), ref(u8rng));
          generate(kernel.begin(), kernel.end(), ref(u8rng));
          generate(bias.begin(), bias.end(), ref(s32rng));
          fill(accumulators.begin(), accumulators.end(), 0);

          ASSERT_NE(
              *max_element(input.cbegin(), input.cend()),
              *min_element(input.cbegin(), input.cend()));
          ASSERT_NE(
              *max_element(kernel.cbegin(), kernel.cend()),
              *min_element(kernel.cbegin(), kernel.cend()));

          fill(packedWeights.begin(), packedWeights.end(), 0xA5);

          usize num_zero_points_padded = channels() + 8;
          vector<u8> kernel_zero_points(
              num_zero_points_padded, 0);
          if (per_channel) {
            generate(
                kernel_zero_points.begin(),
                kernel_zero_points.begin() + channels(),
                ref(u8rng));
          }

          pytorch_pack_q8dw_w(
              kernelHeight(),
              kernelWidth(),
              channels(),
              cr(),
    #if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
              inputZeroPoint(),
              kernel_zero_points.data(),
    #endif
              kernel.data(),
              bias.data(),
              packedWeights.data());

          for (usize i = 0;
               i < kernelSize() + (width() * subsampling() - 1) * kernelHeight();
               i++) {
            indirectInput[i] = inputPtr + i * inputStride();
          }
          shuffle(indirectInput.begin(), indirectInput.end(), rng);

          for (usize x = 0; x < width(); x++) {
            for (usize c = 0; c < channels(); c++) {
              i32 acc = bias[c];
              for (usize kx = 0; kx < kernelWidth(); kx++) {
                for (usize ky = 0; ky < kernelHeight(); ky++) {
                  acc += (i32(indirectInput
                                      [(x * subsampling() + kx) * kernelHeight() +
                                       ky][c]) -
                          i32(inputZeroPoint())) *
                      (i32(
                           kernel[(c * kernelHeight() + ky) * kernelWidth() + kx]) -
                       i32(kernel_zero_points[c]));
                }
              }
              accumulators[x * channels() + c] = acc;
            }
          }
          const i32 accumulatorsMin =
              *min_element(accumulators.cbegin(), accumulators.cend());
          const i32 accumulatorsMax =
              *max_element(accumulators.cbegin(), accumulators.cend());
          const u32 accumulatorsRange =
              u32(accumulatorsMax) - u32(accumulatorsMin);
          ASSERT_NE(0, accumulatorsRange);

          const double outputScale = accumulatorsRange >= 256
              ? double(accumulatorsRange) / 255.0
              : 1.00001;
          const u8 outputZeroPoint = u8(max(
              min(
                  lrint(
                      127.5 -
                      0.5 * double(accumulatorsMin + accumulatorsMax) /
                          outputScale),
                  long(u8::max)),
              long(u8::min)));

          vector<float> requantization_scales(num_zero_points_padded, 1.0f / float(outputScale));
          if (per_channel) {
            auto f32rng =
                bind(uniform_real_distribution<float>(1, 5), rng);
            auto scale_generator = [&]() -> float {return (f32rng()/outputScale);};
            generate(
                requantization_scales.begin(),
                requantization_scales.end(),
                ref(scale_generator));
          }
          const union pytorch_qnnp_conv_quantization_params quantizationParams =
              pytorch_qnnp_compute_conv_quantization_params(
                  inputZeroPoint(),
                  kernel_zero_points.data(),
                  requantization_scales.data(),
                  outputZeroPoint,
                  qmin(),
                  qmax());
          const union pytorch_qnnp_fp32_requantization_params
              scalarRequantizationParams =
                  pytorch_qnnp_compute_scalar_fp32_requantization_params(
                      requantization_scales.data(), outputZeroPoint, qmin(), qmax());

          q8dwconv(
              channels(),
              width(),
              indirectInput.data(),
              packedWeights.data(),
              output.data(),
              kernelHeight() * subsampling() * sizeof(void*),
              (outputStride() - channels()) * sizeof(u8),
              &quantizationParams);

          for (usize x = 0; x < width(); x++) {
            for (usize c = 0; c < channels(); c++) {
    #if defined(__arm__) || defined(_M_ARM)
              const u8 referenceOutput = pytorch_qnnp_fp32_requantize_magic(
                  accumulators[x * channels() + c], scalarRequantizationParams, c);
    #else
              const u8 referenceOutput = pytorch_qnnp_fp32_requantize(
                  accumulators[x * channels() + c], scalarRequantizationParams, c);
    #endif
              const double scaledAccumulator =
                  accumulators[x * channels() + c] * requantization_scales[c] +
                  double(outputZeroPoint);
              const double clampedAccumulator = max(
                  min(scaledAccumulator, double(qmax())), double(qmin()));
              ASSERT_NEAR(
                  clampedAccumulator, double(output[x * outputStride() + c]), 0.6)
                  << "x = " << x << ", channel = " << c;
              ASSERT_EQ(
                  u32(referenceOutput),
                  u32(output[x * outputStride() + c]))
                  << "x = " << x << ", channel = " << c;
            }
          }
        }
        */
    }
    
    pub fn test(&self, 
        q8dwconv:    PyTorchQ8DwConvMpUKernelFunction,
        per_channel: bool)  {
        let per_channel: bool = per_channel.unwrap_or(false);

        todo!();
        /*
            ASSERT_EQ(25, kernelSize())
            << "only 5x5 microkernel is currently supported";

        random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> input(
            (kernelSize() + (width() * subsampling() - 1) * kernelHeight() - 1) *
                inputStride() +
            channels() + 8);
        vector<u8> kernel(channels() * kernelSize());
        vector<u8, AlignedAllocator<u8, 32>> packedWeights(
            (kernelSize() + sizeof(i32) / sizeof(u8)) * packedChannels());
        vector<i32> bias(packedChannels());
        vector<i32> accumulators(width() * channels());
        vector<i32> mpAcc(width() * packedChannels());
        vector<u8> output((width() - 1) * outputStride() + channels());
        vector<const u8*> indirectInput(
            kernelSize() + (width() * subsampling() - 1) * kernelHeight());

        const u8* inputPtr = input.data() + 8;

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(input.begin(), input.end(), ref(u8rng));
          generate(kernel.begin(), kernel.end(), ref(u8rng));
          generate(bias.begin(), bias.end(), ref(s32rng));
          fill(accumulators.begin(), accumulators.end(), 0);
          fill(mpAcc.begin(), mpAcc.end(), 0xA5A55A5A);

          ASSERT_NE(
              *max_element(input.cbegin(), input.cend()),
              *min_element(input.cbegin(), input.cend()));
          ASSERT_NE(
              *max_element(kernel.cbegin(), kernel.cend()),
              *min_element(kernel.cbegin(), kernel.cend()));

          fill(packedWeights.begin(), packedWeights.end(), 0xA5);

          usize num_zero_points_padded = channels() + 8;
          vector<u8> kernel_zero_points(num_zero_points_padded, this->kernelZeroPoint_);
          if (per_channel) {
            generate(
                kernel_zero_points.begin(),
                kernel_zero_points.end(),
                ref(u8rng));
          }

          ASSERT_EQ(25, kernelSize())
              << "only 5x5 microkernel is currently supported";
          pytorch_pack_q8dw_w_dilation(
              kernelHeight(),
              kernelWidth(),
              channels(),
              cr(),
              0,
              kernelHeight(),
              0,
              2,
              kernel.data(),
              bias.data(),
              packedWeights.data(),
              true);
          pytorch_pack_q8dw_w_dilation(
              kernelHeight(),
              kernelWidth(),
              channels(),
              cr(),
              0,
              kernelHeight(),
              2,
              4,
              kernel.data(),
              bias.data(),
              packedWeights.data() +
                  (10 + sizeof(i32) / sizeof(u8)) * packedChannels(),
              false);
          pytorch_pack_q8dw_w_dilation(
              kernelHeight(),
              kernelWidth(),
              channels(),
              cr(),
              0,
              kernelHeight(),
              4,
              5,
              kernel.data(),
              bias.data(),
              packedWeights.data() +
                  (20 + sizeof(i32) / sizeof(u8)) * packedChannels(),
              false);
          for (usize i = 0;
               i < kernelSize() + (width() * subsampling() - 1) * kernelHeight();
               i++) {
            indirectInput[i] = inputPtr + i * inputStride();
          }
          shuffle(indirectInput.begin(), indirectInput.end(), rng);

          for (usize x = 0; x < width(); x++) {
            for (usize c = 0; c < channels(); c++) {
              i32 acc = bias[c];
              for (usize kx = 0; kx < kernelWidth(); kx++) {
                for (usize ky = 0; ky < kernelHeight(); ky++) {
                  acc += (i32(indirectInput
                                      [(x * subsampling() + kx) * kernelHeight() +
                                       ky][c]) -
                          i32(inputZeroPoint())) *
                      (i32(
                           kernel[(c * kernelHeight() + ky) * kernelWidth() + kx]) -
                       i32(kernel_zero_points[c]));
                }
              }
              accumulators[x * channels() + c] = acc;
            }
          }
          const i32 accumulatorsMin =
              *min_element(accumulators.cbegin(), accumulators.cend());
          const i32 accumulatorsMax =
              *max_element(accumulators.cbegin(), accumulators.cend());
          const u32 accumulatorsRange =
              u32(accumulatorsMax) - u32(accumulatorsMin);
          ASSERT_NE(0, accumulatorsRange);

          const double outputScale = accumulatorsRange >= 256
              ? double(accumulatorsRange) / 255.0
              : 1.00001;
          const u8 outputZeroPoint = u8(max(
              min(
                  lrint(
                      127.5 -
                      0.5 * double(accumulatorsMin + accumulatorsMax) /
                          outputScale),
                  long(u8::max)),
              long(u8::min)));

          vector<float> requantization_scales(num_zero_points_padded, 1.0f / float(outputScale));
          if (per_channel) {
            auto f32rng =
                bind(uniform_real_distribution<float>(1, 5), rng);
            auto scale_generator = [&]() -> float {return (f32rng()/outputScale);};
            generate(
                requantization_scales.begin(),
                requantization_scales.end(),
                ref(scale_generator));
          }
          const union pytorch_qnnp_conv_quantization_params quantizationParams =
              pytorch_qnnp_compute_conv_quantization_params(
                  inputZeroPoint(),
                  kernel_zero_points.data(),
                  requantization_scales.data(),
                  outputZeroPoint,
                  qmin(),
                  qmax());
          const union pytorch_qnnp_fp32_requantization_params
              scalarRequantizationParams =
                  pytorch_qnnp_compute_scalar_fp32_requantization_params(
                      requantization_scales.data(), outputZeroPoint, qmin(), qmax());

          q8dwconv(
              channels(),
              width(),
              indirectInput.data(),
              packedWeights.data(),
              mpAcc.data(),
              output.data(),
              kernelHeight() * subsampling() * sizeof(void*),
              (outputStride() - channels()) * sizeof(u8),
              &quantizationParams);

          for (usize x = 0; x < width(); x++) {
            for (usize c = 0; c < channels(); c++) {
    #if defined(__arm__) || defined(_M_ARM)
              const u8 referenceOutput = pytorch_qnnp_fp32_requantize_magic(
                  accumulators[x * channels() + c], scalarRequantizationParams, c);
    #else
              const u8 referenceOutput = pytorch_qnnp_fp32_requantize(
                  accumulators[x * channels() + c], scalarRequantizationParams, c);
    #endif
              const double scaledAccumulator =
                  accumulators[x * channels() + c] * requantization_scales[c] +
                  double(outputZeroPoint);
              const double clampedAccumulator = max(
                  min(scaledAccumulator, double(qmax())), double(qmin()));
              ASSERT_NEAR(
                  clampedAccumulator, double(output[x * outputStride() + c]), 0.6)
                  << "x = " << x << ", channel = " << c;
              ASSERT_EQ(
                  u32(referenceOutput),
                  u32(output[x * outputStride() + c]))
                  << "x = " << x << ", channel = " << c;
            }
          }
        }
        */
    }
}
