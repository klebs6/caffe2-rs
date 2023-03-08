crate::ix!();

use crate::{
    OpSchemaCost,
    GradientMakerBase,
    StorageOrder,
    Tensor,
    OperatorDef,
    CPUContext,
    OperatorStorage,
    TensorShape
};

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

register_cpu_operator!{SpatialBN, SpatialBNOp<CPUContext>}

num_inputs!{SpatialBN, (5,7)}

num_outputs!{SpatialBN, (1,5)}

inputs!{SpatialBN, 
    0 => ("X",      "The input 4-dimensional tensor of shape $NCHW$ or $NHWC$ depending on the order parameter."),
    1 => ("scale",  "The scale as a 1-dimensional tensor of size $C$ to be applied to the output."),
    2 => ("bias",   "The bias as a 1-dimensional tensor of size $C$ to be applied to the output."),
    3 => ("mean",   "The running mean (training) or the estimated mean (testing) as a 1-dimensional tensor of size $C$."),
    4 => ("var",    "The running variance (training) or the estimated variance (testing) as a 1-dimensional tensor of size $C$."),
    5 => ("sums",   "*(optional)* Per-channel sums of elements to be used to determine the mean and variance for this batch."),
    6 => ("sumsq",  "*(optional)* Per-channel sum of elements squared per channel to be used to determine the variance for this batch.")
}

outputs!{SpatialBN, 
    0 => ("Y",            "The output 4-dimensional tensor of the same shape as $X$."),
    1 => ("mean",         "The running mean after the spatial BN operator. Must be in-place with the input *mean*. Should not be used for testing."),
    2 => ("var",          "The running variance after the spatial BN operator. Must be in-place with the input *var*. Should not be used for testing."),
    3 => ("saved_mean",   "Saved mean used during training to speed up gradient computation. Should not be used for testing."),
    4 => ("saved_var",    "Saved variance used during training to speed up gradient computation. Should not be used for testing.")
}

args!{SpatialBN, 

    0 => ("epsilon",      
        "*(type: float; default: 1e-5)* The epsilon value to use to avoid division by zero."),

    1 => ("order",        
        "*(type: string; default: NCHW)* Specifies the order of the input data blob, where $N$ is batch size, 
        $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is NHWC."),

    2 => ("momentum",     
        "*(type: float; default: 0.9)* Factor used in computing the running mean and variance. 
        e.g., running_mean = running_mean x momentum + mean x (1 - momentum)"),

    3 => ("num_batches",  
        "*(type: int; default: 1)* Specifies the number of batches to apply normalization on. 
        Requires specifying the optional sums and sumsq inputs that provide statistics across multiple 
        batches from which mean and variance can be determined.")
}

arg_is_test!{SpatialBN, "*(type: int; default: 0)* If set to nonzero, run spatial batch normalization in test mode."}

inherit_onnx_schema!{SpatialBN, "BatchNormalization"}

allow_inplace!{SpatialBN,   vec![(0, 0), (5, 3), (6, 4)]}

enforce_inplace!{SpatialBN, vec![(3, 1), (4, 2)]}

cost_inference_function!{SpatialBN, CostInferenceForSpatialBN}

tensor_inference_function!{SpatialBN,

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
            ArgumentHelper helper(def);
            bool is_test = helper.GetSingleArgument<int>(OpSchema::Arg_IsTest, 0);

            if (!is_test) {
                vector<TensorShape> out;
                StorageOrder order = StringToStorageOrder(
                    helper.GetSingleArgument<string>("order", "NCHW"));
                const TensorShape& X = in[0];
                const int C =
                    (order == StorageOrder::NCHW ? X.dims(1)
                     : X.dims(X.dims_size() - 1));

                out.push_back(in[0]);
                TensorShape meanvar_tp =
                    CreateTensorShape(vector<int>{C}, TensorProto::FLOAT);
                out.push_back(meanvar_tp); // RUNNING_MEAN
                out.push_back(meanvar_tp); // RUNNING_MEAN
                out.push_back(meanvar_tp); // SAVED_MEAN
                out.push_back(meanvar_tp); // SAVED_VAR
                return out;
            } else {
                return vector<TensorShape>{in[0]};
            }
        */
    }
}

input_tags!{
    SpatialBNOp {
        Input,
        Scale,
        Bias,
        EstMean,
        EstVar,
        BatchMeanSum,
        BatchVarSum
    }
}

output_tags!{
    SpatialBNOp {
        Output,
        RunningMean,
        RunningVar,
        SavedMean,
        SavedInvStd
    }
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

impl SpatialBNGradientOp<CPUContext> {

    #[inline] pub fn compute_scale_bias_gradients_and_fused_params<T>(
        &mut self,
        n:       i32,
        c:       i32,
        hxW:     i32,
        dY:      *const T,
        x:       *const T,
        scale:   *const T,
        mean:    *const T,
        rstd:    *const T,
        dscale:  *mut T,
        dbias:   *mut T,
        alpha:   *mut T,
        beta:    *mut T,
        gamma:   *mut T,
        scratch: *mut T) 
    {
        todo!();
        /*
            ConstEigenVectorArrayMap<T> scale_arr(scale, C);
          ConstEigenVectorArrayMap<T> mean_arr(mean, C);
          ConstEigenVectorArrayMap<T> rstd_arr(rstd, C);
          EigenVectorArrayMap<T> dscale_arr(dscale, C);
          EigenVectorArrayMap<T> dbias_arr(dbias, C);
          EigenVectorArrayMap<T> alpha_arr(alpha, C);
          EigenVectorArrayMap<T> beta_arr(beta, C);
          EigenVectorArrayMap<T> gamma_arr(gamma, C);
          math::Set<T, CPUContext>(C, T(0), dscale, &context_);
          math::Set<T, CPUContext>(C, T(0), dbias, &context_);
          if (order_ == StorageOrder::NCHW) {
            ConstEigenArrayMap<T> dY_arr(dY, HxW, N * C);
            ConstEigenArrayMap<T> X_arr(X, HxW, N * C);
            for (int i = 0; i < N; ++i) {
              for (int j = 0; j < C; ++j) {
                const int c = i * C + j;
                dscale_arr(j) +=
                    (dY_arr.col(c) * (X_arr.col(c) - mean_arr(j)) * rstd_arr(j)).sum();
                dbias_arr(j) += dY_arr.col(c).sum();
              }
            }
          } else {
            const int outer_size = N * HxW;
            ConstEigenArrayMap<T> dY_arr(dY, C, outer_size);
            ConstEigenArrayMap<T> X_arr(X, C, outer_size);
            for (int i = 0; i < outer_size; ++i) {
              dscale_arr += dY_arr.col(i) * (X_arr.col(i) - mean_arr) * rstd_arr;
              dbias_arr += dY_arr.col(i);
            }
          }
          const T inv_nhw = T(1) / static_cast<T>(N * HxW);
          alpha_arr = scale_arr * rstd_arr;
          beta_arr = dscale_arr * rstd_arr;
          gamma_arr = alpha_arr * (mean_arr * beta_arr - dbias_arr) * inv_nhw;
          beta_arr *= -alpha_arr * inv_nhw;
        */
    }

    #[inline] pub fn compute_xgradient<T>(
        &mut self,
        n:      i32,
        c:      i32,
        hxW:    i32,
        dY:     *const T,
        x:      *const T,
        alpha:  *const T,
        beta:   *const T,
        gamma:  *const T,
        dX:     *mut T) {
        todo!();
        /*
            ConstEigenVectorArrayMap<T> alpha_arr(alpha, C);
          ConstEigenVectorArrayMap<T> beta_arr(beta, C);
          ConstEigenVectorArrayMap<T> gamma_arr(gamma, C);
          if (order_ == NCHW) {
            const int stride = C * HxW;
            const T* dY_ptr = dY;
            const T* X_ptr = X;
            T* dX_ptr = dX;
            for (int i = 0; i < N; ++i) {
              EigenArrayMap<T>(dX_ptr, HxW, C) =
                  (ConstEigenArrayMap<T>(dY_ptr, HxW, C).rowwise() *
                       alpha_arr.transpose() +
                   ConstEigenArrayMap<T>(X_ptr, HxW, C).rowwise() *
                       beta_arr.transpose())
                      .rowwise() +
                  gamma_arr.transpose();
              dY_ptr += stride;
              X_ptr += stride;
              dX_ptr += stride;
            }
          } else {
            EigenArrayMap<T>(dX, C, N * HxW) =
                (ConstEigenArrayMap<T>(dY, C, N * HxW).colwise() * alpha_arr +
                 ConstEigenArrayMap<T>(X, C, N * HxW).colwise() * beta_arr)
                    .colwise() +
                gamma_arr;
          }
        */
    }

    #[inline] pub fn compute_multi_batch_scale_bias_gradients_and_fused_params<T>(
        &mut self,
        n:          i32,
        c:          i32,
        hxW:        i32,
        scale:      *const T,
        mean:       *const T,
        rstd:       *const T,
        dscale_sum: *const T,
        dbias_sum:  *const T,
        dscale:     *mut T,
        dbias:      *mut T,
        alpha:      *mut T,
        beta:       *mut T,
        gamma:      *mut T) 
    {
        todo!();
        /*
            ConstEigenVectorArrayMap<T> scale_arr(scale, C);
          ConstEigenVectorArrayMap<T> mean_arr(mean, C);
          ConstEigenVectorArrayMap<T> rstd_arr(rstd, C);
          EigenVectorArrayMap<T> dscale_arr(dscale, C);
          EigenVectorArrayMap<T> dbias_arr(dbias, C);
          EigenVectorArrayMap<T> alpha_arr(alpha, C);
          EigenVectorArrayMap<T> beta_arr(beta, C);
          EigenVectorArrayMap<T> gamma_arr(gamma, C);
          const T inv_num_batches = T(1) / static_cast<T>(num_batches_);
          math::Scale<T, T, CPUContext>(
              C, inv_num_batches, dscale_sum, dscale, &context_);
          math::Scale<T, T, CPUContext>(
              C, inv_num_batches, dbias_sum, dbias, &context_);
          const T inv_nhw = T(1) / static_cast<T>(N * HxW);
          alpha_arr = scale_arr * rstd_arr;
          beta_arr = dscale_arr * rstd_arr;
          gamma_arr = alpha_arr * (mean_arr * beta_arr - dbias_arr) * inv_nhw;
          beta_arr *= -alpha_arr * inv_nhw;
        */
    }

}

///------------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SpatialBNGradientOp<Context> {

    storage: OperatorStorage,
    context: Context,

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

register_cpu_operator!{
    SpatialBNGradient, 
    SpatialBNGradientOp<CPUContext>
}

register_gradient!{
    SpatialBN, 
    GetSpatialBNGradient
}

num_inputs!{SpatialBNGradient, (5,7)}

num_outputs!{SpatialBNGradient, 3}

allow_inplace!{SpatialBNGradient, vec![(5, 1), (6, 2)]}

input_tags!{
    SpatialBNGradientOp {
        Input,
        Scale,
        OutputGrad,
        SavedMean,
        SavedInvStd,
        AggregateScaleGrad,
        AggregateBiasGrad
    }
}

output_tags!{
    SpatialBNGradientOp {
        InputGrad,
        ScaleGrad,
        BiasGrad
    }
}

#[inline] pub fn cost_inference_for_spatialBN(
    def: &OperatorDef, 
    input: &Vec<TensorShape>) -> OpSchemaCost 
{
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<4>(def, in);
      ArgumentHelper helper(def);
      auto order =
          StringToStorageOrder(helper.GetSingleArgument<string>("order", "NCHW"));
      const TensorShape X = in[0];
      const int C =
          (order == StorageOrder::NCHW ? X.dims(1) : X.dims(X.dims_size() - 1));
      cost.params_bytes = 2 * C * sizeof(float);
      return cost;
    */
}

/**
  | Spatial batch normalization's gradient,
  | depending on the various input sizes,
  | is a bit more complex than usual gradient
  | operators.
  |
  */
pub struct GetSpatialBNGradient { }

impl GetGradientDefs for GetSpatialBNGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // Check if we are in training or testing mode.
        const bool is_test =
            ArgumentHelper::GetSingleArgument(def_, OpSchema::Arg_IsTest, 0);
        const int num_batches =
            ArgumentHelper::GetSingleArgument(def_, "num_batches", 1);
        const std::vector<string> grad_outputs = {GI(0), GI(1), GI(2)};
        std::vector<string> grad_inputs;
        if (is_test) {
          // This is in testing mode. The operator should have five inputs:
          //     X, scale, bias, estimated_mean, estimated_variance
          // The gradient inputs are:
          //     X, scale, dY, estimated_mean, estimated_variance
          CAFFE_ENFORCE_EQ(def_.input_size(), 5);
          CAFFE_ENFORCE_EQ(def_.output_size(), 1);
          grad_inputs = std::vector<std::string>{I(0), I(1), GO(0), I(3), I(4)};
        } else if (num_batches > 1) {
          CAFFE_ENFORCE_EQ(def_.input_size(), 7);
          CAFFE_ENFORCE_EQ(def_.output_size(), 5);
          grad_inputs =
              std::vector<std::string>{I(0), I(1), GO(0), O(3), O(4), GI(1), GI(2)};
        } else {
          CAFFE_ENFORCE_EQ(def_.input_size(), 5);
          CAFFE_ENFORCE_EQ(def_.output_size(), 5);
          grad_inputs = std::vector<std::string>{I(0), I(1), GO(0), O(3), O(4)};
        }
        return SingleGradientDef(
            "SpatialBNGradient", "", grad_inputs, grad_outputs);
        */
    }
}
