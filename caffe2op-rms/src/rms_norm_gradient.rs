crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RMSNormGradientOp<Context> {
    storage: OperatorStorage,
    context: Context,
    axis:    i32,
    c2:      Tensor,
}

num_inputs!{RMSNormGradient, 4}

num_outputs!{RMSNormGradient, 3}

impl<Context> RMSNormGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, 1)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& dY = Input(0);
            const auto& X = Input(1);
            const auto& gamma = Input(2);
            const auto& rrms = Input(3);
            const int canonical_axis = X.canonical_axis_index(axis_);
            const int64_t M = X.size_to_dim(canonical_axis);
            const int64_t N = X.size_from_dim(canonical_axis);
            auto* dX = Output(0, X.sizes(), at::dtype<T>());
            auto* dgamma = Output(1, gamma.sizes(), at::dtype<T>());
            auto* dbeta = Output(2, gamma.sizes(), at::dtype<T>());
            const T* dY_data = dY.template data<T>();
            const T* X_data = X.template data<T>();
            const T* gamma_data = gamma.template data<T>();
            const T* rrms_data = rrms.template data<T>();
            T* dX_data = dX->template mutable_data<T>();
            T* dgamma_data = dgamma->template mutable_data<T>();
            T* dbeta_data = dbeta->template mutable_data<T>();

            if (M == 0) {
              math::Set<T, Context>(N, T(0), dgamma_data, &context_);
              math::Set<T, Context>(N, T(0), dbeta_data, &context_);
              return true;
            }

            RMSNormBackward<T>(M, N, dY_data, X_data, gamma_data, rrms_data, dX_data);
            GammaBetaBackward<T>(
                M, N, dY_data, X_data, rrms_data, dgamma_data, dbeta_data);

            return true;
        */
    }
}
