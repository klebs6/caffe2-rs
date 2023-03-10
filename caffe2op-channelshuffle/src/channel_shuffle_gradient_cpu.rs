crate::ix!();

register_cpu_gradient_operator!{
    ChannelShuffleGradient, 
    ChannelShuffleGradientOp<f32, CPUContext>
}

impl ChannelShuffleGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dY = Input(0);

      auto* dX = Output(0, dY.sizes(), at::dtype<float>());
      const int N = dY.dim32(0);
      const int C = dY.dim32(1);
      const int G = group_;
      CAFFE_ENFORCE_EQ(C % G, 0);
      const int K = C / G;
      const int HxW = dY.size_from_dim(2);
      const float* dY_data = dY.data<float>();
      float* dX_data = dX->mutable_data<float>();
      RunChannelShuffleNCHW<float>(N, K, G, HxW, dY_data, dX_data, &context_);
      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dY = Input(0);

      auto* dX = Output(0, dY.sizes(), at::dtype<float>());
      const int ndim = dY.dim();
      const int N = dY.dim32(0);
      const int C = dY.dim32(ndim - 1);
      const int G = group_;
      CAFFE_ENFORCE_EQ(C % G, 0);
      const int K = C / G;
      const int HxW = dY.size_between_dim(0, ndim - 1);
      const float* dY_data = dY.data<float>();
      float* dX_data = dX->mutable_data<float>();
      RunChannelShuffleNHWC<float>(N, K, G, HxW, dY_data, dX_data, &context_);
      return true;
        */
    }
}
