crate::ix!();

impl FloatToHalfOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);

      auto* output = Output(0, input.sizes(), at::dtype<at::Half>());
      const float* data = input.template data<float>();
      at::Half* out = output->template mutable_data<at::Half>();
      auto N = input.numel();

    #ifdef USE_FBGEMM
      // There exists a verion fbgemm::FloatToFloat16_simd which will issue avx-512
      // instructions when possible. However, this actually doesn't give perf
      // benefits, according to benchmarks on T1/T6. Hence we stick to avx2 versions
      // here.
      if (GetCpuId().avx2()) {
        fbgemm::FloatToFloat16_avx2(
            data, reinterpret_cast<fbgemm::float16*>(out), N, clip_);
      } else {
        FloatToFloat16_ref(data, out, N, clip_);
      }
    #else
      FloatToFloat16_ref(data, out, N, clip_);
    #endif

      return true;
        */
    }
}

register_cpu_operator!{FloatToHalf, FloatToHalfOp<CPUContext>}
