crate::ix!();

#[inline] pub fn float_16to_float_ref(
    input:  *const f16,
    out:    *mut f32,
    n:      usize)  {
    
    todo!();
    /*
        for (size_t i = 0; i < N; ++i) {
        out[i] = in[i];
      }
    */
}

impl SparseNormalizeOp<f16, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
          this, Input(INDICES));
        */
    }

    #[inline] pub fn do_run_with_type<SIndex>(&mut self) -> bool {
    
        todo!();
        /*
            const auto* indices = Input(INDICES).template data<SIndex>();
      const auto* paramIn = Input(PARAM).template data<c10::Half>();
      auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<c10::Half>();
      const float kEps = 1e-12f;

      // n: number of sparse embeddings to be normalized
      auto n = Input(INDICES).numel();
      if (n == 0) {
        return true;
      }
      // embedding length, e.g. 32, 64, 128
      auto block_size = Input(PARAM).size_from_dim(1);
      vector<float> row_vec_fp32(block_size);
      auto out_data = row_vec_fp32.data();
      for (int i = 0; i < n; ++i) {
        auto idx = indices[i];
        auto offsetIdx = idx * block_size;
    #ifdef USE_FBGEMM
        if (GetCpuId().avx2()) {
          fbgemm::Float16ToFloat_avx2(
              reinterpret_cast<const fbgemm::float16*>(paramIn + offsetIdx),
              out_data,
              block_size);
        } else {
          Float16ToFloat_ref(paramIn + offsetIdx, out_data, block_size);
        }
    #else
        Float16ToFloat_ref(paramIn + offsetIdx, out_data, block_size);
    #endif
        ConstEigenVectorMap<float> xVec_fp32(row_vec_fp32.data(), block_size);
        float norm = xVec_fp32.template lpNorm<2>();
        if (use_max_norm_ && norm <= norm_) {
          continue;
        }
        auto Y = paramOut + offsetIdx;
        EigenVectorArrayMap<c10::Half>(Y, block_size) *=
            static_cast<float>(norm_ / (norm + kEps));
      }
      return true;
        */
    }
}
