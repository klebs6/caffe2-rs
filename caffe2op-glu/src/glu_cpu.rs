crate::ix!();

impl GluOp<f32, CPUContext> {

    #[inline] pub fn compute_glu(
        &mut self, 
        m:         i32,
        split_dim: i32,
        n:         i32,
        xdata:     *const f32,
        ydata:     *mut f32)  
    {
        todo!();
        /*
            const int xStride = 2 * split_dim * N;
      const int yStride = split_dim * N;
      for (int i = 0; i < M; ++i) {
        const int idx = i * xStride;
        const int idy = i * yStride;
        for (int j = 0; j < split_dim; ++j) {
          const int jN = j * N;
          const int jdx1 = idx + jN;
          const int jdx2 = idx + (j + split_dim) * N;
          const int jdy = idy + jN;
          for (int k = 0; k < N; ++k) {
            const float x1 = Xdata[jdx1 + k];
            const float x2 = Xdata[jdx2 + k];
            Ydata[jdy + k] = x1 * sigmoid(x2);
          }
        }
      }
        */
    }
}
