crate::ix!();

register_cpu_operator!{
    PRelu,
    PReluOp<f32, CPUContext>
}

impl PReluOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
      const auto& W = Input(1);

      auto* Y = Output(0, X.sizes(), at::dtype<float>());
      const auto* Xdata = X.template data<float>();
      const auto* Wdata = W.template data<float>();
      auto* Ydata = Y->template mutable_data<float>();

      const auto C = order_ == StorageOrder::NCHW ? X.size(1) : X.size(X.dim() - 1);
      const auto C_shared = (W.numel() == 1);

      if (!C_shared) {
        CAFFE_ENFORCE_EQ(C, W.numel());
      }

      if (C_shared) {
    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
        // The function is completely pointwise
        runNeonPrelu(Ydata, Xdata, X.size(), Wdata[0]);
    #else
        ConstEigenVectorMap<float> Xvec(Xdata, X.numel());
        EigenVectorMap<float> Yvec(Ydata, Y->numel());
        Yvec = Xvec.cwiseMax(0.f) + Xvec.cwiseMin(0.f) * Wdata[0];
    #endif // defined(__ARM_NEON__) || defined(__ARM_NEON)
        return true;
      }

      // non-shared case.
      switch (order_) {
        case StorageOrder::NCHW: {
          const auto N = X.size(0);
          const auto dim = X.size_from_dim(2);

    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
          // Pointwise for each channel
          for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
              runNeonPrelu(
                  Ydata + (n * C + c) * dim,
                  Xdata + (n * C + c) * dim,
                  dim,
                  Wdata[c]);
            }
          }
    #else
          int nc = 0;
          for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
              ConstEigenVectorMap<float> Xvec(Xdata + nc * dim, dim);
              EigenVectorMap<float>(Ydata + nc * dim, dim) =
                  Xvec.cwiseMax(0.f) + Xvec.cwiseMin(0.f) * Wdata[c];
              nc++;
            }
          }
    #endif
          break;
        }
        case StorageOrder::NHWC: {
          // Lay out matrix as (NHW, C) and multiply by C
          const auto NHW = X.numel() / C;
          ConstEigenArrayMap<float> Xmat(Xdata, C, NHW);
          ConstEigenVectorArrayMap<float> Wvec(Wdata, C);
          EigenArrayMap<float> Ymat(Ydata, C, NHW);
          Ymat = (Xmat > 0).select(Xmat, Xmat.colwise() * Wvec);
          break;
        }
        default:
          CAFFE_THROW("Unknown storage order: ", order_);
      }
      return true;
        */
    }
}
