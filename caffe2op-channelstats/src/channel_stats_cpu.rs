crate::ix!();

impl ChannelStatsOp<CPUContext> {
    
    #[inline] pub fn compute_channel_statsNCHW<T: Float>(
        &mut self, 
        n:         i32,
        c:         i32,
        hxW:       i32,
        x:         *const f32,
        sum:       *mut f32,
        sumsq:     *mut f32) -> bool 
    {
        todo!();
        /*
            ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
      for (int i = 0; i < C; ++i) {
        sum[i] = X_arr.col(i).sum();
        sumsq[i] = X_arr.col(i).square().sum();
      }
      for (int i = 1; i < N; ++i) {
        for (int j = 0; j < C; ++j) {
          const int c = i * C + j;
          sum[j] += X_arr.col(c).sum();
          sumsq[j] += X_arr.col(c).square().sum();
        }
      }
      return true;
        */
    }
    
    #[inline] pub fn compute_channel_statsNHWC(
        &mut self, 
        n:      i32,
        c:      i32,
        hxW:    i32,
        x:      *const f32,
        sum:    *mut f32,
        sumsq:  *mut f32) -> bool 
    {
        todo!();
        /*
            ConstEigenArrayMap<float> X_arr(X, C, N * HxW);
      EigenVectorArrayMap<float> sum_arr(sum, C);
      EigenVectorArrayMap<float> sumsq_arr(sumsq, C);
      sum_arr = X_arr.col(0);
      sumsq_arr = X_arr.col(0).square();
      for (int i = 1; i < N * HxW; ++i) {
        sum_arr += X_arr.col(i);
        sumsq_arr += X_arr.col(i).square();
      }
      return true;
        */
    }
}
