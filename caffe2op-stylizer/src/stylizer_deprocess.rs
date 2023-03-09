crate::ix!();

pub struct BRGNCHWCToPackedInt8BGRAStylizerDeprocessOp {
    storage: OperatorStorage,
    context: CPUContext,
}

impl BRGNCHWCToPackedInt8BGRAStylizerDeprocessOp {

    /// Expect this many channels as input
    const kInputChannels: i32 = 3;

    /// Expect this many channels as output
    const kOutputChannels: i32 = 4;
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        const auto& mean = Input(1);

        CAFFE_ENFORCE(X.dim() == 4);
        const int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
        // Assume BGR or BGRA
        CAFFE_ENFORCE(mean.numel() == kInputChannels);
        CAFFE_ENFORCE(C == kInputChannels);
        // RGB
        auto* Y = Output(0, {N, H, W, kOutputChannels}, at::dtype<uint8_t>());

        runBatch(
            N,
            C,
            H,
            W,
            X.data<float>(),
            mean.data<float>(),
            Y->template mutable_data<uint8_t>());

        return true;
        */
    }
    
    #[inline] pub fn run_batch(&mut self, 
        n:            i32,
        c:            i32,
        h:            i32,
        w:            i32,
        input:        *const f32,
        mean_channel: *const f32,
        output:       *mut u8)  {
        
        todo!();
        /*
            int planeSize = H * W;

        for (int n = 0; n < N; ++n) {
          auto curInput = input + n * kInputChannels * planeSize;
          auto curOutput = output + n * kOutputChannels * planeSize;

    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
          runCPUNeon(H, W, curInput, meanChannel, curOutput);
    #else
          runCPU(H, W, curInput, meanChannel, curOutput);
    #endif //  defined(__ARM_NEON__) || defined(__ARM_NEON)
        }
        */
    }
    
    #[cfg(not(arm_neon))]
    #[inline] pub fn runcpu(&mut self, 
        h:            i32,
        w:            i32,
        input:        *const f32,
        mean_channel: *const f32,
        output:       *mut u8)  {
        
        todo!();
        /*
            int planeSize = H * W;

        for (int point = 0; point < planeSize; ++point) {
          for (int c = 0; c < kInputChannels; ++c) {
            uint8_t v = clamped_cast<uint8_t>(
                input[c * planeSize + point] + meanChannel[c]);
            output[point * kOutputChannels + c] = v;
          }

          // alpha
          output[point * kOutputChannels + (kOutputChannels - 1)] =
              uint8_t::max;
        }
        */
    }
    
    #[cfg(arm_neon)]
    #[inline] pub fn run_cpu_neon(&mut self, 
        h:            i32,
        w:            i32,
        input:        *const f32,
        mean_channel: *const f32,
        output:       *mut u8)  {

        todo!();
        /*
            // Vectorized load parameters:

        // We load in chunks of this size
        constexpr int kLoadUnit = sizeof(float32x4_t);
        constexpr int kLoadFloats = (sizeof(float32x4_t) / sizeof(float));

        // We store in chunks of this size
        constexpr int kStoreUnit = sizeof(uint8x8x4_t);

        // The vector portion loads this many f32 pixels at a time (8)
        constexpr int kLoadPixels = 2 * kLoadFloats;

        float mean[kInputChannels] = {
            meanChannel[0], meanChannel[1], meanChannel[2]};
        int planeSize = H * W;

        // Vectorized portion
        int point = 0;

        // If the slice is not aligned, then we have to use the
        // un-vectorized version
        bool isAligned = isPointerAligned(input, kLoadUnit) &&
            isPointerAligned(output, kStoreUnit) &&
            // Because we are reading from input at offsets of planeSize,
            // planeSize has to be an even multiple of kLoadUnit
            (planeSize % kLoadUnit == 0);

        // What portion the vectorized loop will handle
        int limit = isAligned ? (planeSize / kLoadPixels) * kLoadPixels : 0;

        for (; point < limit; point += kLoadPixels) {
          // Load 8 f32 pixels from each channel; loading 16 involves
          // register spills it seems
          float32x4_t inputc0_0 =
              vld1q_f32_aligned(input + 0 * planeSize + point + 0 * kLoadFloats);
          float32x4_t inputc0_1 =
              vld1q_f32_aligned(input + 0 * planeSize + point + 1 * kLoadFloats);

          float32x4_t inputc1_0 =
              vld1q_f32_aligned(input + 1 * planeSize + point + 0 * kLoadFloats);
          float32x4_t inputc1_1 =
              vld1q_f32_aligned(input + 1 * planeSize + point + 1 * kLoadFloats);

          float32x4_t inputc2_0 =
              vld1q_f32_aligned(input + 2 * planeSize + point + 0 * kLoadFloats);
          float32x4_t inputc2_1 =
              vld1q_f32_aligned(input + 2 * planeSize + point + 1 * kLoadFloats);

          addMeanAndClamp(inputc0_0, mean[0]);
          addMeanAndClamp(inputc0_1, mean[0]);
          uint8x8_t u8_c0 = convertNarrowAndPack(inputc0_0, inputc0_1);

          addMeanAndClamp(inputc1_0, mean[1]);
          addMeanAndClamp(inputc1_1, mean[1]);
          uint8x8_t u8_c1 = convertNarrowAndPack(inputc1_0, inputc1_1);

          addMeanAndClamp(inputc2_0, mean[2]);
          addMeanAndClamp(inputc2_1, mean[2]);
          uint8x8_t u8_c2 = convertNarrowAndPack(inputc2_0, inputc2_1);

          // This is the alpha channel
          uint8x8_t u8_c3 = vdup_n_u8(uint8_t::max);

          // We now have 8 bytes of each channel in a separate vector
          // Write BGRA interleaved to output
          uint8x8x4_t u8_out = {{ u8_c0, u8_c1, u8_c2, u8_c3 }};
          vst4_u8_aligned(output + kOutputChannels * point, u8_out);
        }

        // Epilogue: non-vectorized remainder
        for (; point < planeSize; ++point) {
          for (int c = 0; c < kInputChannels; ++c) {
            uint8_t v =
                clamped_cast<uint8_t>(input[c * planeSize + point] + mean[c]);
            output[point * kOutputChannels + c] = v;
          }

          // alpha
          output[point * kOutputChannels + (kOutputChannels - 1)] =
              uint8_t::max;
        }
        */
    }
}
