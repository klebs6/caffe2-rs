crate::ix!();

register_cpu_operator!{
    ChannelShuffle, 
    ChannelShuffleOp<f32, CPUContext>
}

impl ChannelShuffleOp<f32, CPUContext> {

    #[inline] pub fn run_on_device_with_order_nchw(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

      auto* Y = Output(0, X.sizes(), at::dtype<float>());
      const int N = X.dim32(0);
      const int C = X.dim32(1);
      const int G = group_;
      CAFFE_ENFORCE_EQ(C % G, 0);
      const int K = C / G;
      const int HxW = X.size_from_dim(2);
      const float* X_data = X.data<float>();
      float* Y_data = Y->mutable_data<float>();
      RunChannelShuffleNCHW<float>(N, G, K, HxW, X_data, Y_data, &context_);
      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nhwc(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

      auto* Y = Output(0, X.sizes(), at::dtype<float>());
      const int ndim = X.dim();
      const int N = X.dim32(0);
      const int C = X.dim32(ndim - 1);
      const int G = group_;
      CAFFE_ENFORCE_EQ(C % G, 0);
      const int K = C / G;
      const int HxW = X.size_between_dim(0, ndim - 1);
      const float* X_data = X.data<float>();
      float* Y_data = Y->mutable_data<float>();
      RunChannelShuffleNHWC<float>(N, G, K, HxW, X_data, Y_data, &context_);
      return true;
        */
    }
}
