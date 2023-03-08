crate::ix!();

register_cpu_gradient_operator!{
    PReluGradient, 
    PReluGradientOp<f32, CPUContext>
}

impl PReluGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& Y = Input(0);
      auto& dY = Input(1);
      auto& X = Input(2);
      auto& W = Input(3);

      CAFFE_ENFORCE(&Y != &X, "Cannot backpropagate through an in-place PReLU");

      DCHECK_EQ(dY.numel(), Y.numel());
      auto* dX = Output(0, Y.sizes(), at::dtype<float>());
      auto* dW = Output(1, W.sizes(), at::dtype<float>());

      const auto C = order_ == StorageOrder::NCHW ? X.size(1) : X.size(X.dim() - 1);
      const auto C_shared = (W.numel() == 1);

      const float* Ydata = Y.data<float>();
      const float* dYdata = dY.data<float>();
      const float* Xdata = X.data<float>();
      const float* Wdata = W.data<float>();
      float* dXdata = dX->template mutable_data<float>();
      float* dWdata = dW->template mutable_data<float>();

      // non-shared case.
      switch (order_) {
        case StorageOrder::NCHW: {
          const auto dim = X.size_from_dim(2);
          const auto div_factor = C_shared ? C : 1;
          for (auto c = 0; c < W.numel(); ++c) {
            dWdata[c] = 0;
          }

          for (int i = 0; i < Y.numel(); ++i) {
            if (Xdata[i] <= 0) {
              int c = (i / dim) % C / div_factor;
              dWdata[c] += dYdata[i] * Xdata[i];
            }
          }

          for (int i = 0; i < Y.numel(); ++i) {
            if (Xdata[i] > 0) {
              dXdata[i] = dYdata[i];
            } else {
              int c = (i / dim) % C / div_factor;
              dXdata[i] = Wdata[c] * dYdata[i];
            }
          }
          break;
        }
        case StorageOrder::NHWC: {
          const auto NHW = X.numel() / C;
          ConstEigenVectorArrayMap<float> Wvec(Wdata, W.numel());
          EigenVectorArrayMap<float> dWvec(dWdata, dW->numel());

          ConstEigenArrayMap<float> Ymat(Ydata, C, NHW);
          ConstEigenArrayMap<float> dYmat(dYdata, C, NHW);
          ConstEigenArrayMap<float> Xmat(Xdata, C, NHW);
          EigenArrayMap<float> dXmat(dXdata, C, NHW);

          if (C_shared) {
            dXmat = (Xmat > 0).select(dYmat, dYmat * Wdata[0]);
            dWdata[0] =
                (Xmat > 0)
                    .select(
                        Xmat.cwiseMin(0.0f), // zero gradients on the 'if' path.
                        dYmat * Xmat)
                    .sum();
          } else {
            dXmat = (Xmat > 0).select(dYmat, dYmat.colwise() * Wvec);
            dWvec = (Xmat > 0)
                        .select(
                            Xmat.cwiseMin(0.0f), // zero gradients on the 'if' path.
                            dYmat * Xmat)
                        .rowwise()
                        .sum();
          }
          break;
        }
        default:
          CAFFE_THROW("Unknown storage order: ", order_);
      }

      return true;
        */
    }
}
