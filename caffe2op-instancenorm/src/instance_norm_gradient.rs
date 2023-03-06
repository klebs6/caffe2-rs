crate::ix!();

pub struct InstanceNormGradientOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    epsilon: f32,
    order:   StorageOrder,
    mean:    Tensor,
    rstd:    Tensor,
    ds:      Tensor,
    db:      Tensor,
    c1:      Tensor,
    c2:      Tensor,
    c3:      Tensor,
    ones:    Tensor,

    phantom: PhantomData<T>,
}

num_inputs!{InstanceNormGradient, (4,6)}

num_outputs!{InstanceNormGradient, 3}

register_gradient!{InstanceNorm, GetInstanceNormGradient}

input_tags!{
    InstanceNormGradientOp {
        Input,
        Scale,
        Bias,
        OutputGrad,
        Mean,
        Rstd
    }
}

output_tags!{
    InstanceNormGradientOp {
        InputGrad,
        ScaleGrad,
        BiasGrad
    }
}

impl<T,Context> InstanceNormGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
          OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5),
          order_(StringToStorageOrder(
                  this->template GetSingleArgument<string>("order", "NCHW"))) 

              CAFFE_ENFORCE_GE(epsilon_, 0, "Must pass a nonnegative epsilon.");
          CAFFE_ENFORCE_NE(
              order_,
              StorageOrder::UNKNOWN,
              "order should be either \"NCHW\" or \"NHWC\".");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        const auto& gamma = Input(SCALE);
        const auto& dY = Input(OUTPUT_GRAD);
        const int ndim = X.dim();
        const int64_t N = X.dim(0);
        const int64_t C = order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(ndim - 1);
        const int64_t HxW = X.numel() / (N * C);
        CAFFE_ENFORCE_EQ(gamma.numel(), C);
        const T* dY_data = dY.template data<T>();
        const T* X_data = X.template data<T>();
        const T* gamma_data = gamma.template data<T>();
        const T* mean_data = nullptr;
        const T* rstd_data = nullptr;
        CAFFE_ENFORCE_GE(InputSize(), 4);
        CAFFE_ENFORCE_LE(InputSize(), 6);
        if (InputSize() == 6) {
          const auto& mean = Input(MEAN);
          const auto& rstd = Input(RSTD);
          mean_data = mean.template data<T>();
          rstd_data = rstd.template data<T>();
        } else {
          ReinitializeTensor(
              &mean_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
          ReinitializeTensor(
              &rstd_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
          ComputeMoments(
              N,
              C,
              HxW,
              X_data,
              mean_.template mutable_data<T>(),
              rstd_.template mutable_data<T>());
          mean_data = mean_.template data<T>();
          rstd_data = rstd_.template data<T>();
        }

        auto* dX = Output(INPUT_GRAD, X.sizes(), at::dtype<T>());
        auto* dgamma = Output(SCALE_GRAD, gamma.sizes(), at::dtype<T>());
        auto* dbeta = Output(BIAS_GRAD, gamma.sizes(), at::dtype<T>());
        T* dX_data = dX->template mutable_data<T>();
        T* dgamma_data = dgamma->template mutable_data<T>();
        T* dbeta_data = dbeta->template mutable_data<T>();

        switch (order_) {
          case StorageOrder::NCHW: {
            return RunOnDeviceWithOrderNCHW(
                N,
                C,
                HxW,
                dY_data,
                X_data,
                mean_data,
                rstd_data,
                gamma_data,
                dX_data,
                dgamma_data,
                dbeta_data);
          }
          case StorageOrder::NHWC: {
            return RunOnDeviceWithOrderNHWC(
                N,
                C,
                HxW,
                dY_data,
                X_data,
                mean_data,
                rstd_data,
                gamma_data,
                dX_data,
                dgamma_data,
                dbeta_data);
          }
          default: {
            CAFFE_THROW("Unknown storage order: ", order_);
          }
        }
        */
    }
}
