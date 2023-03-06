crate::ix!();

impl<T, Context> LambdaRankNdcgGradientOp<T, Context> {

    #[inline] pub fn run_f32_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& y = Input(Y);
      auto& sids = Input(SESSION_LENS);
      auto& dy_cache = Input(DY_CACHE);
      auto& dLoss = Input(DLOSS);

      CAFFE_ENFORCE(y.dim() == 1);
      CAFFE_ENFORCE(dy_cache.dim() == 1);
      CAFFE_ENFORCE(dy_cache.numel() > 0);
      CAFFE_ENFORCE(y.numel() == dy_cache.numel());

      const auto* session_lengths = sids.template data<int>();
      CAFFE_ENFORCE(dLoss.numel() == sids.numel());

      ConstEigenVectorArrayMap<float> dy_cache_vec(
          dy_cache.template data<float>(), dy_cache.numel());
      auto* dy = Output(DY, {dy_cache.numel()}, at::dtype<float>());
      EigenVectorArrayMap<float> dy_vec(
          dy->template mutable_data<float>(), dy->numel());
      auto multiplier = dLoss.template data<float>();
      int count = 0;
      for (int j = 0; j < sids.numel(); j++) {
        dy_vec.segment(count, session_lengths[j]) =
            multiplier[j] * dy_cache_vec.segment(count, session_lengths[j]);
        count += session_lengths[j];
      }
      return true;
        */
    }
}

register_cpu_operator!{LambdaRankNdcg, LambdaRankNdcgOp<float, CPUContext>}
register_cpu_operator!{LambdaRankNdcgGradient, LambdaRankNdcgGradientOp<float, CPUContext>}

