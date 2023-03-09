crate::ix!();

impl SparseNormalizeOp<f32, CPUContext> {

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
      const auto* paramIn = Input(PARAM).template data<float>();
      auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();
      const float kEps = 1e-12f;

      // n: number of sparse embeddings to be normalized
      auto n = Input(INDICES).numel();
      if (n == 0) {
        return true;
      }

      // embedding length, e.g. 32, 64, 128
      auto block_size = Input(PARAM).size_from_dim(1);
      for (int i = 0; i < n; ++i) {
        auto idx = indices[i];
        auto offsetIdx = idx * block_size;
        ConstEigenVectorMap<float> xVec(paramIn + offsetIdx, block_size);
        auto norm = xVec.template lpNorm<2>();

        if (use_max_norm_ && norm <= norm_) {
          continue;
        }

        math::Scale(
            block_size,
            norm_ / (norm + kEps),
            paramOut + offsetIdx,
            paramOut + offsetIdx,
            &context_);
      }
      return true;
        */
    }
}
