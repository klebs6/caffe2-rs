crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SpatialBNGradientOp<Context> {
    storage:     OperatorStorage,
    context:     Context,
    epsilon:     f64,
    order:       StorageOrder,
    num_batches: i32,
    alpha:       Tensor,
    beta:        Tensor,
    gamma:       Tensor,
    ones:        Tensor,

    /*
      | Input: X, scale, dY, mean, variance,
      | dscale, dbias
      | 
      | Output: dX, dscale, dbias
      |
      */
}

impl<Context> SpatialBNGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(double, "epsilon", epsilon_, 1e-5),
            order_(StringToStorageOrder(
                    this->template GetSingleArgument<string>("order", "NCHW"))),
                    OP_SINGLE_ARG(int, "num_batches", num_batches_, 1) 

                        CAFFE_ENFORCE_NE(
                            order_,
                            StorageOrder::UNKNOWN,
                            "order should be either \"NCHW\" or \"NHWC\".");
            CAFFE_ENFORCE(InputSize() == 5 || InputSize() == 7);
            CAFFE_ENFORCE_EQ(OutputSize(), 3);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(INPUT);
            const auto& dY = Input(OUTPUT_GRAD);
            const auto& scale = Input(SCALE);
            const auto& mean = Input(SAVED_MEAN);
            const auto& rstd = Input(SAVED_INV_STD);
            const int ndim = X.dim();
            CAFFE_ENFORCE_GE(ndim, 3);
            const int N = X.dim32(0);
            const int C =
                (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1));
            const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
            const int HxW =
                std::accumulate(
                    X_dims.cbegin() + 1, X_dims.cend(), 1, std::multiplies<int>()) /
                C;
            CAFFE_ENFORCE_EQ(scale.numel(), C);
            CAFFE_ENFORCE_EQ(mean.numel(), C);
            CAFFE_ENFORCE_EQ(rstd.numel(), C);

            auto* dX = Output(INPUT_GRAD, X.sizes(), at::dtype<T>());
            at::IntArrayRef dscale_sizes, dbias_sizes;
            if (num_batches_ == 1) {
              dscale_sizes = scale.sizes();
              dbias_sizes = scale.sizes();
            } else {
              const auto& dscale_sum = Input(AGGREGATE_SCALE_GRAD);
              const auto& dbias_sum = Input(AGGREGATE_BIAS_GRAD);
              // Note: previously there was alias check to decide whether to call
              // ResizeLike or not, since we only call Resize when the size does not
              // match the size of cached Tensor, this check is not necessary
              dscale_sizes = dscale_sum.sizes();
              dbias_sizes = dbias_sum.sizes();
            }
            auto* dscale = Output(SCALE_GRAD, dscale_sizes, at::dtype<T>());
            auto* dbias = Output(BIAS_GRAD, dbias_sizes, at::dtype<T>());
            const T* X_data = X.template data<T>();
            const T* dY_data = dY.template data<T>();
            const T* scale_data = scale.template data<T>();
            const T* mean_data = mean.template data<T>();
            const T* rstd_data = rstd.template data<T>();
            T* dX_data = dX->template mutable_data<T>();
            T* dscale_data = dscale->template mutable_data<T>();
            T* dbias_data = dbias->template mutable_data<T>();

            if (N == 0) {
              math::Set<T, Context>(C, T(0), dscale_data, &context_);
              math::Set<T, Context>(C, T(0), dbias_data, &context_);
              return true;
            }
            ReinitializeTensor(
                &alpha_, {C}, at::dtype<T>().device(Context::GetDeviceType()));
            ReinitializeTensor(
                &beta_, {C}, at::dtype<T>().device(Context::GetDeviceType()));
            ReinitializeTensor(
                &gamma_, {C}, at::dtype<T>().device(Context::GetDeviceType()));
            T* alpha_data = alpha_.template mutable_data<T>();
            T* beta_data = beta_.template mutable_data<T>();
            T* gamma_data = gamma_.template mutable_data<T>();
            if (num_batches_ > 1) {
              const auto& dscale_sum = Input(AGGREGATE_SCALE_GRAD);
              const auto& dbias_sum = Input(AGGREGATE_BIAS_GRAD);
              ComputeMultiBatchScaleBiasGradientsAndFusedParams<T>(
                  N,
                  C,
                  HxW,
                  scale_data,
                  mean_data,
                  rstd_data,
                  dscale_sum.template data<T>(),
                  dbias_sum.template data<T>(),
                  dscale_data,
                  dbias_data,
                  alpha_data,
                  beta_data,
                  gamma_data);
            } else {
              ComputeScaleBiasGradientsAndFusedParams<T>(
                  N,
                  C,
                  HxW,
                  dY_data,
                  X_data,
                  scale_data,
                  mean_data,
                  rstd_data,
                  dscale_data,
                  dbias_data,
                  alpha_data,
                  beta_data,
                  gamma_data,
                  dX_data);
            }
            ComputeXGradient<T>(
                N, C, HxW, dY_data, X_data, alpha_data, beta_data, gamma_data, dX_data);

            return true;
        */
    }
}
