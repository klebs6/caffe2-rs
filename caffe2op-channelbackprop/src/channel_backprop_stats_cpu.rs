crate::ix!();

impl ChannelBackpropStatsOp<CPUContext> {

    #[inline] pub fn run_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
      const auto& dY = Input(OUTPUT_GRAD);
      CAFFE_ENFORCE(X.dim() >= 3 && X.dim() <= 5);
      const int N = X.dim32(0);
      const int C = X.dim32(1);
      const int H = X.dim32(2);
      const int W = X.dim() > 3 ? X.dim32(3) : 1;
      const int D = X.dim() > 4 ? X.dim32(4) : 1;

      const int sampleSize = H * W * D;

      Output(SCALE_GRAD)->Resize(C);
      Output(BIAS_GRAD)->Resize(C);
      auto* dScale = Output(SCALE_GRAD);
      auto* dBias = Output(BIAS_GRAD);

      ConstEigenArrayMap<float> X_arr(X.data<float>(), sampleSize, N * C);
      ConstEigenArrayMap<float> dY_arr(dY.data<float>(), sampleSize, N * C);
      ConstEigenVectorArrayMap<float> mean_arr(Input(SAVED_MEAN).data<float>(), C);
      ConstEigenVectorArrayMap<float> inv_stddev_arr(
          Input(SAVED_INV_STDDEV).data<float>(), C);
      EigenVectorArrayMap<float> dBias_arr(
          dBias->template mutable_data<float>(), C);
      EigenVectorArrayMap<float> dScale_arr(
          dScale->template mutable_data<float>(), C);

      dBias_arr.setZero();
      dScale_arr.setZero();

      for (int nc = 0; nc < N * C; ++nc) {
        int c = nc % C;
        dBias_arr(c) += dY_arr.col(nc).sum();
        dScale_arr(c) +=
            ((X_arr.col(nc) - mean_arr(c)) * inv_stddev_arr(c) * dY_arr.col(nc))
                .sum();
      }
      return true;
        */
    }
}
