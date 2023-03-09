crate::ix!();

/**
  | Applies spatial batch normalization
  | to the input tensor as described in the
  | original paper, [Batch
  | 
  | Normalization: Accelerating Deep
  | Network Training by Reducing Internal
  | Covariate Shift]
  | 
  | (https://arxiv.org/abs/1502.03167).
  | 
  | Be aware, this operator has two different
  | output sets, depending on the value
  | of is_test*. According to the paper,
  | the primary operation of spatial batch
  | normalization is:
  | 
  | $$Y = \frac{X - \mu_x}{\sqrt{\sigma^2_{x}
  | + \epsilon}}*\gamma + b$$
  | 
  | In the equation, $\mu_x$ is the *mean*,
  | $X$ is the input data, $\sigma^2_{x}$
  | is the *var*, $\epsilon$ is *epsilon*,
  | $\gamma$ is the *scale*, $b$ is the *bias*,
  | and $Y$ is the output data.
  | 
  | The *momentum* arg also affects this
  | calculation in the computation of the
  | running mean and variance.
  | 
  | The influence of *momentum* is as follows:
  | 
  | $$running\_mean = running\_mean *
  | momentum + mean (1 - momentum)$$
  | 
  | $$running\_var = running\_var * momentum
  | + var (1 - momentum)$$
  | 
  | Output when is_test = 0 (train mode):
  | *Y, mean, var, saved_mean, saved_var*
  | 
  | Output when is_test = 1 (test mode):
  | *Y*
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_batch_norm_op.cc
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_batch_norm_op.h
  |
  */
pub struct SpatialBNOp<Context> {
    storage:      OperatorStorage,
    context:      Context,
    is_test:      bool,
    epsilon:      f64,
    momentum:     f32,
    order:        StorageOrder,
    num_batches:  i32,
    alpha:        Tensor,
    beta:         Tensor,
}

impl<Context> SpatialBNOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(bool, OpSchema::Arg_IsTest, is_test_, false),
            OP_SINGLE_ARG(double, "epsilon", epsilon_, 1e-5),
            OP_SINGLE_ARG(float, "momentum", momentum_, 0.9f),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<std::string>("order", "NCHW"))),
            OP_SINGLE_ARG(int, "num_batches", num_batches_, 1) 

            CAFFE_ENFORCE_NE(
                order_,
                StorageOrder::UNKNOWN,
                "order should be either \"NCHW\" or \"NHWC\".");
            CAFFE_ENFORCE(
                (is_test_ && OutputSize() == 1) || (!is_test_ && OutputSize() == 5));
            CAFFE_ENFORCE_GT(epsilon_, 0);
            CAFFE_ENFORCE_GE(momentum_, 0);
            CAFFE_ENFORCE_LE(momentum_, 1);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>() -> bool {
        todo!();
        /*
            const auto& X = Input(INPUT);
            const auto& scale = Input(SCALE);
            const auto& bias = Input(BIAS);

            const int ndim = X.dim();
            CAFFE_ENFORCE_GE(ndim, 2);
            const int N = X.dim32(0);
            const int C =
                (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1));
            const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
            CAFFE_ENFORCE_NE(C, 0);
            const int HxW =
                std::accumulate(
                    X_dims.cbegin() + 1, X_dims.cend(), 1, std::multiplies<int>()) /
                C;
            CAFFE_ENFORCE_EQ(scale.numel(), C);
            CAFFE_ENFORCE_EQ(bias.numel(), C);

            auto* Y = Output(OUTPUT, X.sizes(), at::dtype<T>());
            const T* X_data = X.template data<T>();
            const T* scale_data = scale.template data<T>();
            const T* bias_data = bias.template data<T>();
            T* Y_data = Y->template mutable_data<T>();
            ReinitializeTensor(
                &alpha_, {C}, at::dtype<T>().device(Context::GetDeviceType()));
            ReinitializeTensor(
                &beta_, {C}, at::dtype<T>().device(Context::GetDeviceType()));
            T* alpha_data = alpha_.template mutable_data<T>();
            T* beta_data = beta_.template mutable_data<T>();
            if (is_test_) {
              if (N == 0) {
                return true;
              }
              const auto& mean = Input(EST_MEAN);
              const auto& var = Input(EST_VAR);
              CAFFE_ENFORCE_EQ(mean.numel(), C);
              CAFFE_ENFORCE_EQ(var.numel(), C);
              ComputeFusedParam<T>(
                  C,
                  scale_data,
                  bias_data,
                  mean.template data<T>(),
                  var.template data<T>(),
                  alpha_data,
                  beta_data);
            } else {
              auto* saved_mean = Output(SAVED_MEAN, {C}, at::dtype<T>());
              auto* saved_rstd = Output(SAVED_INV_STD, {C}, at::dtype<T>());
              T* saved_mean_data = saved_mean->template mutable_data<T>();
              T* saved_rstd_data = saved_rstd->template mutable_data<T>();

              // Enforce Alias
              CAFFE_ENFORCE(
                  IsInputOutputAlias(3, 1), "Input 3 and Output 1 should be alias.");
              CAFFE_ENFORCE(
                  IsInputOutputAlias(4, 2), "Input 4 and Output 2 should be alias.");

              Tensor* running_mean = nullptr;
              Tensor* running_var = nullptr;
              const auto& mean = Input(EST_MEAN);
              const auto& var = Input(EST_VAR);
              if (mean.numel() != C) {
                running_mean = Output(RUNNING_MEAN, {C}, at::dtype<T>());
                C10_LOG_EVERY_MS(WARNING, 1000)
                    << "[Depreacated] Running mean is not initialized in "
                       "SpatialBatchNorm Op";
                math::Set<T, Context>(
                    C, T(0), running_mean->template mutable_data<T>(), &context_);
              } else {
                running_mean = Output(RUNNING_MEAN, {C}, at::dtype<T>());
              }
              if (var.numel() != C) {
                running_var = Output(RUNNING_VAR, {C}, at::dtype<T>());
                math::Set<T, Context>(
                    C, T(0), running_var->template mutable_data<T>(), &context_);
                C10_LOG_EVERY_MS(WARNING, 1000)
                    << "[Deprecated] Running variance is not initialized in "
                       "SpatialBatchNorm Op";
              } else {
                running_var = Output(RUNNING_VAR, {C}, at::dtype<T>());
              }

              T* running_mean_data = running_mean->template mutable_data<T>();
              T* running_var_data = running_var->template mutable_data<T>();
              if (N == 0) {
                math::Set<T, Context>(C, T(0), saved_mean_data, &context_);
                math::Set<T, Context>(C, T(0), saved_rstd_data, &context_);
                return true;
              }
              if (num_batches_ > 1) {
                const auto& batch_mean_sum = Input(BATCH_MEAN_SUM);
                const auto& batch_var_sum = Input(BATCH_VAR_SUM);
                CAFFE_ENFORCE_EQ(batch_mean_sum.numel(), C);
                CAFFE_ENFORCE_EQ(batch_var_sum.numel(), C);
                ComputeBatchMoments<T>(
                    N,
                    C,
                    HxW,
                    batch_mean_sum.template data<T>(),
                    batch_var_sum.template data<T>(),
                    saved_mean_data,
                    saved_rstd_data);
              } else {
                if (order_ == StorageOrder::NCHW) {
                  const std::array<int, 3> X_dims_arr = {N, C, HxW};
                  const std::array<int, 3> Y_dims_arr = {1, C, 1};
                  math::Moments<T, Context>(
                      3,
                      X_dims_arr.data(),
                      Y_dims_arr.data(),
                      X_data,
                      saved_mean_data,
                      saved_rstd_data,
                      &context_);
                } else {
                  const std::array<int, 2> X_dims_arr = {N * HxW, C};
                  const std::array<int, 2> Y_dims_arr = {1, C};
                  math::Moments<T, Context>(
                      2,
                      X_dims_arr.data(),
                      Y_dims_arr.data(),
                      X_data,
                      saved_mean_data,
                      saved_rstd_data,
                      &context_);
                }
              }
              ComputeRunningMomentsAndFusedParam<T>(
                  C,
                  num_batches_ * N * HxW,
                  scale_data,
                  bias_data,
                  saved_mean_data,
                  saved_rstd_data,
                  running_mean_data,
                  running_var_data,
                  saved_rstd_data,
                  alpha_data,
                  beta_data);
            }
            if (order_ == StorageOrder::NCHW) {
              math::AffineChannel<T, Context, StorageOrder::NCHW>(
                  N, C, HxW, X_data, alpha_data, beta_data, Y_data, &context_);
            } else {
              math::AffineChannel<T, Context, StorageOrder::NHWC>(
                  N, C, HxW, X_data, alpha_data, beta_data, Y_data, &context_);
            }

            return true;
        */
    }

    #[inline] pub fn compute_fused_param<T>(
        &mut self,
        c:       i32,
        scale:   *const T,
        bias:    *const T,
        mean:    *const T,
        var:     *const T,
        alpha:   *mut T,
        beta:    *mut T) 
    {
        todo!();
        /*
            EigenVectorArrayMap<T> alpha_arr(alpha, C);
            EigenVectorArrayMap<T> beta_arr(beta, C);
            alpha_arr = ConstEigenVectorArrayMap<T>(scale, C) *
                (ConstEigenVectorArrayMap<T>(var, C) + static_cast<T>(epsilon_))
                    .rsqrt();
            beta_arr = ConstEigenVectorArrayMap<T>(bias, C) -
                alpha_arr * ConstEigenVectorArrayMap<T>(mean, C);
        */
    }

    #[inline] pub fn compute_batch_moments<T>(
        &mut self,
        n:              i32,
        c:              i32,
        hxW:            i32,
        batch_mean_sum: *const T,
        batch_var_sum:  *const T,
        mean:           *mut T,
        var:            *mut T) 
    {
        todo!();
        /*
            const T scale = T(1) / static_cast<T>(num_batches_ * N * HxW);
            EigenVectorArrayMap<T> mean_arr(mean, C);
            EigenVectorArrayMap<T> var_arr(var, C);
            mean_arr = ConstEigenVectorArrayMap<T>(batch_mean_sum, C) * scale;
            var_arr = ConstEigenVectorArrayMap<T>(batch_var_sum, C) * scale -
                mean_arr.square();
        */
    }


    #[inline] pub fn compute_running_moments_and_fused_param<T>(
        &mut self, 
        c:             i32,
        reduce_size:   i32,
        scale:         *const T,
        bias:          *const T,
        mean:          *const T,
        var:           *const T,
        running_mean:  *mut T,
        running_var:   *mut T,
        rstd:          *mut T,
        alpha:         *mut T,
        beta:          *mut T) 
    {
        todo!();
        /*
            const T a = T(1) - static_cast<T>(momentum_);
            const T b = static_cast<T>(momentum_);
            const T unbias_scale = reduce_size == 1
                ? std::numeric_limits<T>::infinity()
                : static_cast<T>(reduce_size) / static_cast<T>(reduce_size - 1);
            math::Axpby<T, T, Context>(C, a, mean, b, running_mean, &context_);
            math::Axpby<T, T, Context>(
                C, a * unbias_scale, var, b, running_var, &context_);
            math::InvStd<T, Context>(C, static_cast<T>(epsilon_), var, rstd, &context_);
            EigenVectorArrayMap<T> alpha_arr(alpha, C);
            EigenVectorArrayMap<T> beta_arr(beta, C);
            alpha_arr = ConstEigenVectorArrayMap<T>(scale, C) *
                ConstEigenVectorArrayMap<T>(rstd, C);
            beta_arr = ConstEigenVectorArrayMap<T>(bias, C) -
                alpha_arr * ConstEigenVectorArrayMap<T>(mean, C);
        */
    }
}
