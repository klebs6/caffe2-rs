crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ChannelShuffleOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    order:   StorageOrder,
    group:   i32,
    phantom: PhantomData<T>,
}

num_inputs!{ChannelShuffle, 1}

num_outputs!{ChannelShuffle, 1}

identical_type_and_shape!{ChannelShuffle}

inherit_onnx_schema!{ChannelShuffle}

impl<T,Context> ChannelShuffleOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
          order_(StringToStorageOrder(
                  this->template GetSingleArgument<std::string>("order", "NCHW"))),
                  OP_SINGLE_ARG(int, "group", group_, 1) 

                      CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return order_ == StorageOrder::NCHW ? RunOnDeviceWithOrderNCHW()
                                            : RunOnDeviceWithOrderNHWC();
        */
    }
}

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ChannelShuffleGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,

    order:   StorageOrder,
    group:   i32,
    phantom: PhantomData<T>,
}

num_inputs!{ChannelShuffleGradient, 1}

num_outputs!{ChannelShuffleGradient, 1}

identical_type_and_shape!{ChannelShuffleGradient}

impl<T,Context> ChannelShuffleGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<std::string>("order", "NCHW"))),
            OP_SINGLE_ARG(int, "group", group_, 1) 

        CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return order_ == StorageOrder::NCHW ? RunOnDeviceWithOrderNCHW()
                                            : RunOnDeviceWithOrderNHWC();
        */
    }
}

#[inline] pub fn run_channel_shuffleNCHW<T>(
    n:         i32,
    g:         i32,
    k:         i32,
    hxW:       i32,
    x:         *const T,
    y:         *mut T,
    context:   *mut CPUContext) 
{
    todo!();
    /*
        const int stride = G * K * HxW;
      for (int i = 0; i < N; ++i) {
        if (G < K) {
          for (int j = 0; j < G; ++j) {
            math::CopyMatrix<T, CPUContext>(
                K, HxW, X + j * K * HxW, HxW, Y + j * HxW, G * HxW, context);
          }
        } else {
          for (int j = 0; j < K; ++j) {
            math::CopyMatrix<T, CPUContext>(
                G, HxW, X + j * HxW, K * HxW, Y + j * G * HxW, HxW, context);
          }
        }
        X += stride;
        Y += stride;
      }
    */
}


#[inline] pub fn run_channel_shuffleNHWC<T>(
    n:         i32,
    g:         i32,
    k:         i32,
    hxW:       i32,
    x:         *const T,
    y:         *mut T,
    context:   *mut CPUContext) 
{
    todo!();
    /*
        const std::array<std::int64_t, 2> dims = {G, K};
      const std::array<std::int32_t, 2> axes = {1, 0};
      const int M = N * HxW;
      const int C = G * K;
      for (int i = 0; i < M; ++i) {
        math::Transpose<std::int64_t, T, CPUContext>(
            2, dims.data(), axes.data(), X, Y, context);
        X += C;
        Y += C;
      }
    */
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

register_cpu_operator!{
    ChannelShuffle, 
    ChannelShuffleOp<f32, CPUContext>
}

register_cpu_gradient_operator!{
    ChannelShuffleGradient, 
    ChannelShuffleGradientOp<f32, CPUContext>
}

pub struct GetChannelShuffleGradient;

impl GetGradientDefs for GetChannelShuffleGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ChannelShuffleGradient",
            "",
            std::vector<std::string>{GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{ChannelShuffle, GetChannelShuffleGradient}
