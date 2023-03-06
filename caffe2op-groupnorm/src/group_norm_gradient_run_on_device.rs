crate::ix!();

impl<T, Context> GroupNormGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "group", group_, 32),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<std::string>("order", "NCHW"))) 

        CAFFE_ENFORCE_NE(
            order_,
            StorageOrder::UNKNOWN,
            "order should be either \"NCHW\" or \"NHWC\".");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dY = Input(OUTPUT_GRAD);
        const auto& X = Input(INPUT);
        const auto& gamma = Input(GAMMA);
        const auto& beta = Input(BETA);
        const auto& mu = Input(MU);
        const auto& rsig = Input(INV_SIGMA);
        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
        const int HxW = X.numel() / (N * C);
        CAFFE_ENFORCE_EQ(C % group_, 0);
        CAFFE_ENFORCE_EQ(gamma.numel(), C);
        CAFFE_ENFORCE_EQ(beta.numel(), C);
        const int G = group_;
        const int K = C / G;
        auto* dX = Output(INPUT_GRAD, X.sizes(), at::dtype<T>());
        auto* dgamma = Output(GAMMA_GRAD, gamma.sizes(), at::dtype<T>());
        auto* dbeta = Output(BETA_GRAD, beta.sizes(), at::dtype<T>());
        if (order_ == StorageOrder::NCHW) {
          return RunOnDeviceWithOrderNCHW(
              N,
              G,
              K,
              HxW,
              dY.template data<T>(),
              X.template data<T>(),
              mu.template data<T>(),
              rsig.template data<T>(),
              gamma.template data<T>(),
              dX->template mutable_data<T>(),
              dgamma->template mutable_data<T>(),
              dbeta->template mutable_data<T>());
        } else {
          return RunOnDeviceWithOrderNHWC(
              N,
              G,
              K,
              HxW,
              dY.template data<T>(),
              X.template data<T>(),
              mu.template data<T>(),
              rsig.template data<T>(),
              gamma.template data<T>(),
              dX->template mutable_data<T>(),
              dgamma->template mutable_data<T>(),
              dbeta->template mutable_data<T>());
        }
        */
    }
}
