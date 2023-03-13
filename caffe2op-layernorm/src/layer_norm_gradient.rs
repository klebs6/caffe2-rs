crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LayerNormGradientOp<Context> {
    storage:            OperatorStorage,
    context:            Context,
    axis:               i32,
    elementwise_affine: bool,
    ds:                 Tensor,
    db:                 Tensor,
    rstd:               Tensor,
    x_scale:            Tensor,
    bias:               Tensor,
    g_scale:            Tensor,
    ones:               Tensor,
}

num_inputs!{LayerNormGradient, (5,6)}

num_outputs!{LayerNormGradient, (1,3)}

register_cpu_operator!{LayerNormGradient, LayerNormGradientOp<CPUContext>}

impl<Context> LayerNormGradientOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, 1),
            OP_SINGLE_ARG(bool, "elementwise_affine", elementwise_affine_, false)
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
            const auto& dY = Input(0);
            const auto& Y = Input(1);
            const auto& mean = Input(2);
            const auto& sigma = Input(3);
            const auto& X = Input(4);

            const int canonical_axis = X.canonical_axis_index(axis_);
            const int M = X.size_to_dim(canonical_axis);
            const int N = X.size_from_dim(canonical_axis);

            auto* dX = Output(0, X.sizes(), at::dtype<T>());
            ReinitializeTensor(
                &ds_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
            ReinitializeTensor(
                &db_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
            ReinitializeTensor(
                &rstd_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
            ReinitializeTensor(
                &X_scale_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
            ReinitializeTensor(
                &bias_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
            const T* dY_data = dY.template data<T>();
            const T* X_data = X.template data<T>();
            const T* mean_data = mean.template data<T>();
            const T* sigma_data = sigma.template data<T>();
            T* dX_data = dX->template mutable_data<T>();
            T* ds_data = ds_.template mutable_data<T>();
            T* db_data = db_.template mutable_data<T>();
            T* rstd_data = rstd_.template mutable_data<T>();
            T* X_scale_data = X_scale_.template mutable_data<T>();
            T* bias_data = bias_.template mutable_data<T>();

            const T* gamma_data = nullptr;
            T* dgamma_data = nullptr;
            T* dbeta_data = nullptr;
            T* g_scale_data = nullptr;
            if (elementwise_affine_) {
              const auto& gamma = Input(5);
              auto* dgamma = Output(1, gamma.sizes(), at::dtype<T>());
              auto* dbeta = Output(2, gamma.sizes(), at::dtype<T>());
              ReinitializeTensor(
                  &g_scale_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
              gamma_data = gamma.template data<T>();
              dgamma_data = dgamma->template mutable_data<T>();
              dbeta_data = dbeta->template mutable_data<T>();
              g_scale_data = g_scale_.template mutable_data<T>();
            }

            if (M == 0) {
              if (N > 0 && dgamma_data != nullptr) {
                math::Set<T, Context>(N, T(0), dgamma_data, &context_);
              }
              if (N > 0 && dbeta_data != nullptr) {
                math::Set<T, Context>(N, T(0), dbeta_data, &context_);
              }
              return true;
            }

            ComputeInternalGradients<T>(
                M, N, dY_data, X_data, gamma_data, dX_data, ds_data, db_data);
            ComputeFusedParams<T>(
                M,
                N,
                mean_data,
                sigma_data,
                ds_data,
                db_data,
                rstd_data,
                X_scale_data,
                bias_data,
                g_scale_data);
            if (elementwise_affine_) {
              GammaBetaBackward<T>(
                  M,
                  N,
                  dX_data,
                  dY_data,
                  rstd_data,
                  g_scale_data,
                  dgamma_data,
                  dbeta_data);
            }
            LayerNormBackward<T>(
                M,
                N,
                dY_data,
                X_data,
                gamma_data,
                rstd_data,
                X_scale_data,
                bias_data,
                dX_data);

            return true;
        */
    }
}

