crate::ix!();

/**
 | Computes layer normalization as described in
 | https://arxiv.org/pdf/1607.06450.pdf.
 |
 | Given an input vector x \in [a_0, a_1,
 | ...,a_{k-1}, a_k, ..., a_{n-1}], this op treats
 | dimensions a_k through a_{n-1} as feature vectors. 
 |
 | For each feature vector, the op contains the mean
 | and standard deviation. 
 |
 | Then, it returns the normalized values (with
 | respect to the feature vector).
 |
 | Note that this op does not contain the scale an
 | bias terms described in the paper. 
 |
 | Simply follow this op with an FC op to add
 | those. Concretely, this op implements:
 |
 | h = \frac{1}{\sigma}(a - \mu)
 | where \mu = \frac{1}{H}\sum_{i=1}^{H} a_i
 | and \sigma = \sqrt{\frac{1}{H}\sum_{i=1}^{H}(a_i - \mu)^2}
 | where H is the number of hidden units (i.e. product of dimensions from 'axis'
 | to the end.)
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LayerNormOp<Context> {
    storage:             OperatorStorage,
    context:             Context,
    axis:                i32,
    epsilon:             f32,
    elementwise_affine:  bool,
    scale:               Tensor, //{Context::GetDeviceType()};
    bias:                Tensor, //{Context::GetDeviceType()};
}

num_inputs!{LayerNorm, (1,3)}

num_outputs!{LayerNorm, 3}

inputs!{LayerNorm, 
    0 => ("input",  "Input tensor which layer normalization will be applied to"),
    1 => ("gamma",  "scale tensor for elementwise_affine, the shape should be the same as the dimensions of X begin from axis"),
    2 => ("beta",   "bias tensor for elementwise_affine, the shape should be the same as the dimensions of X begin from axis")
}

outputs!{LayerNorm, 
    0 => ("output", "Normalized values"),
    1 => ("mean",   "Mean values for each feature vector"),
    2 => ("stddev", "Standard deviations for each feature vector")
}

args!{LayerNorm, 
    0 => ("axis",               "(int) default to 1; Describes axis of the inputs. Defaults to one because the 0th axis most likely describes the batch size"),
    1 => ("epsilon",            "(float) default to 0.001. Small value to be added to the stdev when dividing out by that value. This prevents division by zero."),
    2 => ("elementwise_affine", "(bool) default to False; If true, this op will do affine transformation after normalization.")
}

tensor_inference_function!{LayerNorm, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      std::vector<TensorShape> out(3);
      auto input_dims_long = GetDimsVector(in[0]);
      std::vector<int> input_dims(
          input_dims_long.begin(), input_dims_long.end());
      out[0] = CreateTensorShape(input_dims, TensorProto::FLOAT);

      ArgumentHelper helper(def);

      auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
      const auto canonical_axis =
          canonical_axis_index_(axis, in[0].dims().size());
      std::vector<int> stat_dims(
          input_dims.begin(), input_dims.begin() + canonical_axis);
      stat_dims.push_back(1);
      out[1] = CreateTensorShape(stat_dims, TensorProto::FLOAT);
      out[2] = CreateTensorShape(stat_dims, TensorProto::FLOAT);
      return out;
    } */
}

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

impl LayerNormOp<CPUContext> {

    #[inline] pub fn compute_sigma_and_fused_params<T>(
        &mut self,
        n:       i32,
        eps:     f32,
        mean:    *const T,
        var:     *const T,
        sigma:   *mut T,
        scale:   *mut T,
        bias:    *mut T) 
    {
        todo!();
        /*
            ConstEigenVectorArrayMap<T> var_arr(var, N);
          EigenVectorArrayMap<T> sigma_arr(sigma, N);
          sigma_arr = var_arr + static_cast<T>(eps);
          math::Rsqrt<T, CPUContext>(N, sigma, scale, &context_);
          math::Mul<T, CPUContext>(N, scale, sigma, sigma, &context_);
          EigenVectorArrayMap<T>(bias, N) = -ConstEigenVectorArrayMap<T>(scale, N) *
              ConstEigenVectorArrayMap<T>(mean, N);
        */
    }

    #[inline] pub fn layer_norm_forward<T>(
        &mut self, 
        m:      i32,
        n:      i32,
        x:      *const T,
        scale:  *const T,
        bias:   *const T,
        gamma:  *const T,
        beta:   *const T,
        y:      *mut T) 
    {
        todo!();
        /*
            ConstEigenArrayMap<T> X_arr(X, N, M);
          ConstEigenVectorArrayMap<T> scale_arr(scale, M);
          ConstEigenVectorArrayMap<T> bias_arr(bias, M);
          EigenArrayMap<T> Y_arr(Y, N, M);
          if (gamma != nullptr && beta != nullptr) {
            ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
            ConstEigenVectorArrayMap<T> beta_arr(beta, N);
            Y_arr = (((X_arr.rowwise() * scale_arr.transpose()).rowwise() +
                      bias_arr.transpose())
                         .colwise() *
                     gamma_arr)
                        .colwise() +
                beta_arr;
          } else {
            CAFFE_ENFORCE(gamma == nullptr);
            CAFFE_ENFORCE(beta == nullptr);
            Y_arr = (X_arr.rowwise() * scale_arr.transpose()).rowwise() +
                bias_arr.transpose();
          }
        */
    }
}

///----------------------------------------
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

impl LayerNormGradientOp<CPUContext> {

    #[inline] pub fn compute_internal_gradients<T>(
        &mut self,
        m:       i32,
        n:       i32,
        dY:      *const T,
        x:       *const T,
        gamma:   *const T,
        d_yxX:   *mut T,
        ds:      *mut T,
        db:      *mut T) 
    {
        todo!();
        /*
            math::Mul<T, CPUContext>(M * N, dY, X, dYxX, &context_);
          ConstEigenArrayMap<T> dYxX_arr(dYxX, N, M);
          ConstEigenArrayMap<T> dY_arr(dY, N, M);
          if (gamma != nullptr) {
            ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
            for (int i = 0; i < M; ++i) {
              ds[i] = (dYxX_arr.col(i) * gamma_arr).sum();
              db[i] = (dY_arr.col(i) * gamma_arr).sum();
            }
          } else {
            EigenVectorArrayMap<T>(ds, M) = dYxX_arr.colwise().sum();
            EigenVectorArrayMap<T>(db, M) = dY_arr.colwise().sum();
          }
        */
    }

    #[inline] pub fn compute_fused_params<T>(
        &mut self,
        m:         i32,
        n:         i32,
        mean:      *const T,
        sigma:     *const T,
        ds:        *const T,
        db:        *const T,
        rstd:      *mut T,
        x_scale:   *mut T,
        bias:      *mut T,
        g_scale:   *mut T) 
    {
        todo!();
        /*
            const T scale = T(1) / static_cast<T>(N);
          ConstEigenVectorArrayMap<T> mean_arr(mean, M);
          ConstEigenVectorArrayMap<T> ds_arr(ds, M);
          ConstEigenVectorArrayMap<T> db_arr(db, M);
          EigenVectorArrayMap<T> rstd_arr(rstd, M);
          EigenVectorArrayMap<T> X_scale_arr(X_scale, M);
          rstd_arr = ConstEigenVectorArrayMap<T>(sigma, M).inverse();
          X_scale_arr = (db_arr * mean_arr - ds_arr) * rstd_arr.cube() * scale;
          EigenVectorArrayMap<T>(bias, M) =
              -X_scale_arr * mean_arr - db_arr * rstd_arr * scale;
          if (g_scale != nullptr) {
            EigenVectorArrayMap<T>(g_scale, M) = -rstd_arr * mean_arr;
          }
        */
    }

    #[inline] pub fn layer_norm_backward<T>(
        &mut self,
        m:         i32,
        n:         i32,
        dY:        *const T,
        x:         *const T,
        gamma:     *const T,
        dY_scale:  *const T,
        x_scale:   *const T,
        bias:      *const T,
        dX:        *mut T) 
    {
        todo!();
        /*
            ConstEigenArrayMap<T> dY_arr(dY, N, M);
          ConstEigenArrayMap<T> X_arr(X, N, M);
          EigenArrayMap<T> dX_arr(dX, N, M);
          if (gamma != nullptr) {
            ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
            for (int i = 0; i < M; ++i) {
              dX_arr.col(i) = dY_arr.col(i) * gamma_arr * dY_scale[i] +
                  X_arr.col(i) * X_scale[i] + bias[i];
            }
          } else {
            ConstEigenVectorArrayMap<T> dY_scale_arr(dY_scale, M);
            ConstEigenVectorArrayMap<T> X_scale_arr(X_scale, M);
            ConstEigenVectorArrayMap<T> bias_arr(bias, M);
            dX_arr = (dY_arr.rowwise() * dY_scale_arr.transpose() +
                      X_arr.rowwise() * X_scale_arr.transpose())
                         .rowwise() +
                bias_arr.transpose();
          }
        */
    }

    #[inline] pub fn gamma_beta_backward<T>(
        &mut self,
        m:        i32,
        n:        i32,
        d_yxX:    *const T,
        dY:       *const T,
        rstd:     *const T,
        g_scale:  *const T,
        dgamma:   *mut T,
        dbeta:    *mut T)  
    {
        todo!();
        /*
            math::Set<T, CPUContext>(N, T(0), dgamma, &context_);
          math::Set<T, CPUContext>(N, T(0), dbeta, &context_);
          ConstEigenArrayMap<T> dYxX_arr(dYxX, N, M);
          ConstEigenArrayMap<T> dY_arr(dY, N, M);
          EigenVectorArrayMap<T> dgamma_arr(dgamma, N);
          EigenVectorArrayMap<T> dbeta_arr(dbeta, N);
          for (int i = 0; i < M; ++i) {
            dgamma_arr += dYxX_arr.col(i) * rstd[i] + dY_arr.col(i) * g_scale[i];
            dbeta_arr += dY_arr.col(i);
          }
        */
    }
}

register_cpu_operator!{LayerNorm, LayerNormOp<CPUContext>}

pub struct GetLayerNormGradient;

impl GetGradientDefs for GetLayerNormGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            bool elementwise_affine = false;
            if (ArgumentHelper::HasArgument(Def(), "elementwise_affine")) {
              elementwise_affine = GetArgument(Def(), "elementwise_affine").i();
            }
            if (elementwise_affine) {
              return SingleGradientDef(
                  "LayerNormGradient",
                  "",
                  std::vector<std::string>{GO(0), O(0), O(1), O(2), I(0), I(1)},
                  std::vector<std::string>{GI(0), GI(1), GI(2)});
            } else {
              return SingleGradientDef(
                  "LayerNormGradient",
                  "",
                  std::vector<std::string>{GO(0), O(0), O(1), O(2), I(0)},
                  std::vector<std::string>{GI(0)});
            }
        */
    }
}

register_gradient!{LayerNorm, GetLayerNormGradient}
