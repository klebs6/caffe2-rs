crate::ix!();

impl<T, Context> GroupNormOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "group", group_, 32),
            OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<std::string>("order", "NCHW"))),
            OP_SINGLE_ARG(bool, OpSchema::Arg_IsTest, is_test_, true) 

        CAFFE_ENFORCE_NE(
            order_,
            StorageOrder::UNKNOWN,
            "order should be either \"NCHW\" or \"NHWC\".");
        if (!is_test_) {
          CAFFE_ENFORCE_EQ(OutputSize(), 3);
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        const auto& gamma = Input(GAMMA);
        const auto& beta = Input(BETA);
        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
        const size_t HxW = order_ == StorageOrder::NCHW
            ? X.size_from_dim(2)
            : X.size_between_dim(0, ndim - 1);
        CAFFE_ENFORCE_EQ(C % group_, 0);
        CAFFE_ENFORCE_EQ(gamma.numel(), C);
        CAFFE_ENFORCE_EQ(beta.numel(), C);
        const int G = group_;
        const int K = C / G;
        auto* Y = Output(OUTPUT, X.sizes(), at::dtype<T>());
        if (N == 0) {
          return true;
        }
        T* mu_data = nullptr;
        T* rsig_data = nullptr;
        if (OutputSize() == 3) {
          auto* mu = Output(MU, {N, G}, at::dtype<T>());
          auto* rsig = Output(INV_SIGMA, {N, G}, at::dtype<T>());
          mu_data = mu->template mutable_data<T>();
          rsig_data = rsig->template mutable_data<T>();
        } else {
          ReinitializeTensor(
              &mu_, {N, G}, at::dtype<T>().device(Context::GetDeviceType()));
          ReinitializeTensor(
              &rsig_, {N, G}, at::dtype<T>().device(Context::GetDeviceType()));
          mu_data = mu_.template mutable_data<T>();
          rsig_data = rsig_.template mutable_data<T>();
        }
        if (order_ == StorageOrder::NCHW) {
          return RunOnDeviceWithOrderNCHW(
              N,
              G,
              K,
              HxW,
              X.template data<T>(),
              gamma.template data<T>(),
              beta.template data<T>(),
              Y->template mutable_data<T>(),
              mu_data,
              rsig_data);
        } else {
          return RunOnDeviceWithOrderNHWC(
              N,
              G,
              K,
              HxW,
              X.template data<T>(),
              gamma.template data<T>(),
              beta.template data<T>(),
              Y->template mutable_data<T>(),
              mu_data,
              rsig_data);
        }
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(
        &mut self, 
        n:       i32,
        g:       i32,
        k:       i32,
        hxW:     i32,
        x:       *const T,
        gamma:   *const T,
        beta:    *const T,
        y:       *mut T,
        mu:      *mut T,
        rsig:    *mut T) -> bool {

        todo!();
        /*
            const int C = G * K;
        ReinitializeTensor(
            &scale_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
        ReinitializeTensor(
            &bias_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
        T* scale_data = scale_.template mutable_data<T>();
        T* bias_data = bias_.template mutable_data<T>();
        const std::array<int, 2> X_dims = {N * G, K * HxW};
        const std::array<int, 2> Y_dims = {N * G, 1};
        math::Moments<T, Context>(
            2, X_dims.data(), Y_dims.data(), X, mu, rsig, &context_);
        math::InvStd<T, Context>(
            N * G, static_cast<T>(epsilon_), rsig, rsig, &context_);
        ComputeFusedParams(N, G, K, mu, rsig, gamma, beta, scale_data, bias_data);
        GroupNormForwardNCHW(N, C, HxW, X, scale_data, bias_data, Y);
        return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(
        &mut self, 
        n:      i32,
        g:      i32,
        k:      i32,
        hxW:    i32,
        x:      *const T,
        gamma:  *const T,
        beta:   *const T,
        y:      *mut T,
        mu:     *mut T,
        rsig:   *mut T) -> bool {

        todo!();
        /*
            const int C = G * K;
        ReinitializeTensor(
            &scale_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
        ReinitializeTensor(
            &bias_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
        T* scale_data = scale_.template mutable_data<T>();
        T* bias_data = bias_.template mutable_data<T>();
        const std::array<int, 4> X_dims = {N, HxW, G, K};
        const std::array<int, 4> Y_dims = {N, 1, G, 1};
        math::Moments<T, Context>(
            4, X_dims.data(), Y_dims.data(), X, mu, rsig, &context_);
        math::InvStd<T, Context>(
            N * G, static_cast<T>(epsilon_), rsig, rsig, &context_);
        ComputeFusedParams(N, G, K, mu, rsig, gamma, beta, scale_data, bias_data);
        GroupNormForwardNHWC(N, C, HxW, X, scale_data, bias_data, Y);
        return true;
        */
    }
}
