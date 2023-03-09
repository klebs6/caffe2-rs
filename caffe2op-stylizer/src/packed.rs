crate::ix!();

pub struct PackedInt8BGRANHWCToNCHWCStylizerPreprocessOp {
    storage: OperatorStorage,
    context: CPUContext,
    ws:      *mut Workspace,
}

impl PackedInt8BGRANHWCToNCHWCStylizerPreprocessOp {

    /// Expect this many channels as input
    pub const kInputChannels: i32 = 4;

    /// Expect this many channels as output
    pub const kOutputChannels: i32 = 3;

    /// We read this much noise per vectorized cycle
    pub const kNeonNoiseReadSize: i32 = Self::kOutputChannels * 16;

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {

      todo!();
      /*
          : Operator<CPUContext>(operator_def, ws), ws_(ws)
      */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        const auto& mean = Input(1);

        auto* noiseBlob = ws_->CreateBlob("__CAFFE2_STYLIZER_NOISE__");
        auto defaultNoiseSize = OperatorStorage::GetSingleArgument<int>(
            "noise_size", 491 /* prime to avoid artifacts */);

        if (!BlobIsTensorType(*noiseBlob, CPU)) {
          // Initialize random noise on first use.
          // Cache it to maintain temporal consistency.
          auto* t = BlobGetMutableTensor(noiseBlob, CPU);

    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
          // Noise space is larger for vectorized code due to the
          // vectorized load
          initNoiseCPUNeon(t, defaultNoiseSize);
    #else
          initNoiseCPU(t, defaultNoiseSize);
    #endif
        }
        const auto& noise = noiseBlob->template Get<TensorCPU>();
        CAFFE_ENFORCE(noise.numel() >= defaultNoiseSize);

        CAFFE_ENFORCE(X.dim() == 4);
        const int N = X.dim32(0), H = X.dim32(1), W = X.dim32(2), C = X.dim32(3);
        // Assume BGR or BGRA
        CAFFE_ENFORCE(mean.numel() == kOutputChannels);

        CAFFE_ENFORCE(C == kInputChannels);
        auto* Y = Output(0, {N, kOutputChannels, H, W}, at::dtype<float>());

        runBatch(
            N,
            C,
            H,
            W,
            defaultNoiseSize,
            X.data<uint8_t>(),
            mean.data<float>(),
            noise.data<float>(),
            Y->template mutable_data<float>());

        return true;
        */
    }
    
    #[cfg(not(arm_neon))]
    #[inline] pub fn init_noisecpu(&mut self, noise: *mut Tensor, size: i32)  {
        
        todo!();
        /*
            noise->Resize(size);

        math::RandGaussian<float, CPUContext>(
            size,
            0.0,
            OperatorStorage::GetSingleArgument<float>("noise_std", 10.0),
            noise->template mutable_data<float>(),
            &context_);
        */
    }
    
    #[cfg(arm_neon)]
    #[inline] pub fn init_noise_cpu_neon(&mut self, noise: *mut Tensor, size: i32)  {
        
        todo!();
        /*
            // For ARM NEON, we read in multiples of kNeonNoiseReadSize since
        // the inner loop is vectorized. Round up to the next highest
        // multiple of kNeonNoiseReadSize
        size = math::RoundUp(size, kNeonNoiseReadSize) + size;
        noise->Resize(size);

        math::RandGaussian<float, CPUContext>(
            size,
            0.0,
            OperatorStorage::GetSingleArgument<float>("noise_std", 10.0),
            noise->template mutable_data<float>(),
            &context_);
        */
    }
    
    #[inline] pub fn run_batch(&mut self, 
        n:            i32,
        c:            i32,
        h:            i32,
        w:            i32,
        noise_cycle:  i32,
        input:        *const u8,
        mean_channel: *const f32,
        noise:        *const f32,
        output:       *mut f32)  {
        
        todo!();
        /*
            int planeSize = H * W;

        for (int n = 0; n < N; ++n) {
          auto curInput = input + n * kInputChannels * planeSize;
          auto curOutput = output + n * kOutputChannels * planeSize;

    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
          runCPUNeon(H, W, noiseCycle, curInput, meanChannel, noise, curOutput);
    #else
          runCPU(H, W, noiseCycle, curInput, meanChannel, noise, curOutput);
    #endif // defined(__ARM_NEON__) || defined(__ARM_NEON)
        }
        */
    }
    
    #[cfg(not(arm_neon))]
    #[inline] pub fn run_cpu(&mut self, 
        h:            i32,
        w:            i32,
        noise_cycle:  i32,
        input:        *const u8,
        mean_channel: *const f32,
        noise:        *const f32,
        output:       *mut f32)  {

        todo!();
        /*
            int planeSize = H * W;
        int noiseOffset = 0;

        for (int point = 0; point < planeSize; ++point) {
          for (int c = 0; c < kOutputChannels; ++c) {
            float v = (float)input[point * kInputChannels + c];
            output[c * planeSize + point] = v - meanChannel[c] + noise[noiseOffset];

            if (++noiseOffset >= noiseCycle) {
              noiseOffset = 0;
            }
          }
        }
        */
    }
    
    #[cfg(arm_neon)]
    #[inline] pub fn run_cpu_neon(&mut self, 
        h:            i32,
        w:            i32,
        noise_cycle:  i32,
        input:        *const u8,
        mean_channel: *const f32,
        noise:        *const f32,
        output:       *mut f32)  {

        todo!();
        /*
            // Vectorized load parameters:

        // Loop unroll factor
        // FIXME: this doesn't actually unroll; clang has per-loop unroll
        // pragmas but GCC does not
        constexpr int kUnroll = 1;

        // How much data we load for each inner loop
        constexpr int kInnerLoadSize = sizeof(uint8x16x4_t);

        // What we write out
        constexpr int kInnerStoreSize = sizeof(float32x4_t);

        // We load 16 pixels at a time, with 4 channels each
        constexpr int kLoadPixels = kInnerLoadSize / kInputChannels;
        static_assert(kLoadPixels == 16, "unexpected");

        // How many pixels we load per loop
        constexpr int kLoadPixelsPerLoop = kLoadPixels * kUnroll;

        // We need at least this much noise each loop through
        CAFFE_ENFORCE_GE(noiseCycle, kOutputChannels * kLoadPixelsPerLoop);

        int noiseUsed = 0;
        const float* curNoise = noise;

        float mean[kOutputChannels] = {
            meanChannel[0], meanChannel[1], meanChannel[2]};
        int planeSize = H * W;

        // Vectorized portion
        int point = 0;

        // If the slice is not aligned, then we have to use the
        // un-vectorized version
        bool isAligned = isPointerAligned(input, kInnerLoadSize) &&
            isPointerAligned(output, kInnerStoreSize) &&
            // Because we are writing to output at offsets of planeSize,
            // planeSize has to be an even multiple of kInnerStoreSize
            (planeSize % kInnerStoreSize == 0);

        // What portion the vectorized loop will handle
        int limit =
            isAligned ? (planeSize / kLoadPixelsPerLoop) * kLoadPixelsPerLoop : 0;

        for (; point < limit; point += kLoadPixelsPerLoop) {
          // Unroll load/update/store by kUnroll
          for (int j = 0; j < kUnroll; ++j) {
            // We load 16 pixels x 4 channels at a time
            const uint8_t* inputAligned = (const uint8_t*)__builtin_assume_aligned(
                input + (point + j * kLoadPixels) * kInputChannels,
                sizeof(uint8x16x4_t));
            uint8x16x4_t loadV = vld4q_u8(inputAligned);

            // The compiler doesn't want to unroll this when we put it in a
            // loop, and in GCC there's no per-loop unroll pragma, so we do
            // it manually.
            // This seems to involve no register spillage, crossing fingers
            // that it remains that way.
            {
              constexpr int kChannel = 0;
              float32x4_t noise0 = vld1q_f32(curNoise + j * 48 + 0);
              float32x4_t noise1 = vld1q_f32(curNoise + j * 48 + 4);
              float32x4_t noise2 = vld1q_f32(curNoise + j * 48 + 8);
              float32x4_t noise3 = vld1q_f32(curNoise + j * 48 + 12);

              float32x4x4_t outV = to_f32_v4_x4(loadV.val[kChannel]);
              float32x4_t meanV = vdupq_n_f32(mean[kChannel]);
              outV.val[0] = vsubq_f32(outV.val[0], meanV);
              outV.val[1] = vsubq_f32(outV.val[1], meanV);
              outV.val[2] = vsubq_f32(outV.val[2], meanV);
              outV.val[3] = vsubq_f32(outV.val[3], meanV);

              outV.val[0] = vaddq_f32(outV.val[0], noise0);
              outV.val[1] = vaddq_f32(outV.val[1], noise1);
              outV.val[2] = vaddq_f32(outV.val[2], noise2);
              outV.val[3] = vaddq_f32(outV.val[3], noise3);

              float* outputAligned = (float*)__builtin_assume_aligned(
                  &output[kChannel * planeSize + (point + j * kLoadPixels)],
                  sizeof(float32x4_t));

              vst1q_f32(outputAligned + 0, outV.val[0]);
              vst1q_f32(outputAligned + 4, outV.val[1]);
              vst1q_f32(outputAligned + 8, outV.val[2]);
              vst1q_f32(outputAligned + 12, outV.val[3]);
            }

            {
              constexpr int kChannel = 1;
              float32x4_t noise0 = vld1q_f32(curNoise + j * 48 + 16);
              float32x4_t noise1 = vld1q_f32(curNoise + j * 48 + 20);
              float32x4_t noise2 = vld1q_f32(curNoise + j * 48 + 24);
              float32x4_t noise3 = vld1q_f32(curNoise + j * 48 + 28);

              float32x4x4_t outV = to_f32_v4_x4(loadV.val[kChannel]);
              float32x4_t meanV = vdupq_n_f32(mean[kChannel]);
              outV.val[0] = vsubq_f32(outV.val[0], meanV);
              outV.val[1] = vsubq_f32(outV.val[1], meanV);
              outV.val[2] = vsubq_f32(outV.val[2], meanV);
              outV.val[3] = vsubq_f32(outV.val[3], meanV);

              outV.val[0] = vaddq_f32(outV.val[0], noise0);
              outV.val[1] = vaddq_f32(outV.val[1], noise1);
              outV.val[2] = vaddq_f32(outV.val[2], noise2);
              outV.val[3] = vaddq_f32(outV.val[3], noise3);

              float* outputAligned = (float*)__builtin_assume_aligned(
                  &output[kChannel * planeSize + (point + j * kLoadPixels)],
                  sizeof(float32x4_t));

              vst1q_f32(outputAligned + 0, outV.val[0]);
              vst1q_f32(outputAligned + 4, outV.val[1]);
              vst1q_f32(outputAligned + 8, outV.val[2]);
              vst1q_f32(outputAligned + 12, outV.val[3]);
            }

            {
              constexpr int kChannel = 2;
              float32x4_t noise0 = vld1q_f32(curNoise + j * 48 + 32);
              float32x4_t noise1 = vld1q_f32(curNoise + j * 48 + 36);
              float32x4_t noise2 = vld1q_f32(curNoise + j * 48 + 40);
              float32x4_t noise3 = vld1q_f32(curNoise + j * 48 + 44);

              float32x4x4_t outV = to_f32_v4_x4(loadV.val[kChannel]);
              float32x4_t meanV = vdupq_n_f32(mean[kChannel]);
              outV.val[0] = vsubq_f32(outV.val[0], meanV);
              outV.val[1] = vsubq_f32(outV.val[1], meanV);
              outV.val[2] = vsubq_f32(outV.val[2], meanV);
              outV.val[3] = vsubq_f32(outV.val[3], meanV);

              outV.val[0] = vaddq_f32(outV.val[0], noise0);
              outV.val[1] = vaddq_f32(outV.val[1], noise1);
              outV.val[2] = vaddq_f32(outV.val[2], noise2);
              outV.val[3] = vaddq_f32(outV.val[3], noise3);

              float* outputAligned = (float*)__builtin_assume_aligned(
                  &output[kChannel * planeSize + (point + j * kLoadPixels)],
                  sizeof(float32x4_t));

              vst1q_f32(outputAligned + 0, outV.val[0]);
              vst1q_f32(outputAligned + 4, outV.val[1]);
              vst1q_f32(outputAligned + 8, outV.val[2]);
              vst1q_f32(outputAligned + 12, outV.val[3]);
            }
          }

          curNoise += (kLoadPixels * kOutputChannels) * kUnroll;
          noiseUsed += (kLoadPixels * kOutputChannels) * kUnroll;

          if (noiseUsed >= noiseCycle) {
            noiseUsed = 0;
            curNoise = noise + ((curNoise - noise) % noiseCycle);
          }
        }

        // Epilogue: non-vectorized remainder
        for (; point < planeSize; ++point) {
          for (int c = 0; c < kOutputChannels; ++c) {
            float v = (float)input[point * kInputChannels + c];
            output[c * planeSize + point] = v - mean[c] + *curNoise++;
            ++noiseUsed;
          }

          if (noiseUsed >= noiseCycle) {
            noiseUsed = 0;
            curNoise = noise + ((curNoise - noise) % noiseCycle);
          }
        }
        */
    }
}
