crate::ix!();

impl HalfToFloatOp<CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);

      auto* output = Output(0, input.sizes(), at::dtype<float>());
      const at::Half* data = input.template data<at::Half>();
      float* out = output->template mutable_data<float>();
      auto N = input.numel();

    #ifdef USE_FBGEMM
      // Same reasoning of sticking to avx2
      if (GetCpuId().avx2()) {
        fbgemm::Float16ToFloat_avx2(
            reinterpret_cast<const fbgemm::float16*>(data), out, N);
      } else {
        Float16ToFloat_ref(data, out, N);
      }
    #else
      Float16ToFloat_ref(data, out, N);
    #endif

      return true;
        */
    }
}

register_cpu_operator!{HalfToFloat, HalfToFloatOp<CPUContext>}
