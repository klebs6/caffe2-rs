crate::ix!();

impl<Context> LayerNormOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, 1),
            OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5f),
            OP_SINGLE_ARG(bool, "elementwise_affine", elementwise_affine_, false)
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
            const auto& X = Input(0);
            auto* Y = Output(0);
            CAFFE_ENFORCE_GE(X.dim(), 2, "LayerNorm requires input dim >= 2.");
            const int canonical_axis = X.canonical_axis_index(axis_);
            std::vector<int64_t> moments_dims(
                X.sizes().cbegin(), X.sizes().cbegin() + canonical_axis);
            moments_dims.push_back(1);
            auto* mean = Output(1, moments_dims, at::dtype<T>());
            auto* sigma = Output(2, moments_dims, at::dtype<T>());
            const int M = X.size_to_dim(canonical_axis);
            const int N = X.size_from_dim(canonical_axis);
            Y->ResizeLike(X);
            scale_.Resize(M);
            bias_.Resize(M);
            const T* X_data = X.template data<T>();
            T* Y_data = Y->template mutable_data<T>();
            T* mean_data = mean->template mutable_data<T>();
            T* sigma_data = sigma->template mutable_data<T>();
            T* scale_data = scale_.template mutable_data<T>();
            T* bias_data = bias_.template mutable_data<T>();

            if (M == 0) {
              return true;
            }

            const std::array<int, 2> X_dims = {M, N};
            const std::array<int, 2> Y_dims = {M, 1};
            math::Moments<T, Context>(
                2,
                X_dims.data(),
                Y_dims.data(),
                X_data,
                mean_data,
                sigma_data,
                &context_);
            ComputeSigmaAndFusedParams<T>(
                M, epsilon_, mean_data, sigma_data, sigma_data, scale_data, bias_data);
            const T* gamma_data = nullptr;
            const T* beta_data = nullptr;
            if (elementwise_affine_) {
              CAFFE_ENFORCE_EQ(InputSize(), 3);
              const auto& gamma = Input(1);
              const auto& beta = Input(2);
              CAFFE_ENFORCE_EQ(gamma.numel(), N);
              CAFFE_ENFORCE_EQ(beta.numel(), N);
              gamma_data = gamma.template data<T>();
              beta_data = beta.template data<T>();
            }
            LayerNormForward<T>(
                M, N, X_data, scale_data, bias_data, gamma_data, beta_data, Y_data);
            return true;
        */
    }
}

