crate::ix!();

use crate::{
    OperatorStorage,
    Tensor,
    CPUContext,
    StorageOrder,
    GradientMakerBase,
    OperatorDef,
};

#[test] fn instance_norm_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "InstanceNorm",
        ["input", "scale", "bias"],
        ["output"],
        epsilon=1e-5,
    )

    workspace.FeedBlob("input", np.random.randn(2, 1, 3, 3).astype(np.float32))
    print("input:\n", workspace.FetchBlob("input"), "\n")

    workspace.FeedBlob("scale", np.array([1.5]).astype(np.float32))
    print("scale: ", workspace.FetchBlob("scale"))

    workspace.FeedBlob("bias", np.array([1.]).astype(np.float32))
    print("bias: ", workspace.FetchBlob("bias"))

    workspace.RunOperatorOnce(op)
    print("output:\n", workspace.FetchBlob("output"))

    input:
     [[[[ 0.97856593 -1.1832817  -0.2540021 ]
       [-1.3315694  -0.7485018   0.3787225 ]
       [-0.6826597  -1.4637762   0.57116514]]]


     [[[-0.44948956  0.85544354 -0.9315333 ]
       [-0.37202677 -0.22266895 -0.27194235]
       [ 0.4948163  -0.7296504   1.3393803 ]]]]

    scale:  [1.5]
    bias:  [1.]
    output:
     [[[[ 3.5017493  -0.3791256   1.2890853 ]
       [-0.6453266   0.40137637  2.4249308 ]
       [ 0.5195738  -0.8826599   2.7703972 ]]]


     [[[ 0.12639964  2.856744   -0.8821926 ]
       [ 0.28847694  0.60098207  0.49788612]
       [ 2.1021945  -0.45978796  3.869297  ]]]]
    */
}

/**
  | The *InstanceNorm* op applies Instance
  | Normalization over a 4D input as described
  | in [Instance Normalization: The Missing
  | Ingredient for Fast Stylization] (https://arxiv.org/abs/1607.08022).
  | 
  | $$output = \frac{input-\mu_{input}}{\sqrt{\sigma_{input}^2}
  | + \epsilon}*scale + bias$$
  | 
  | Notice, two of the outputs are optional
  | so there are three output cases for this
  | op.
  | 
  | Case 1: output; Case 2: output, saved_mean;
  | Case 3: output, saved_mean, saved_inv_stdev.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.h
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.cc
  |
  */
pub struct InstanceNormOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    epsilon: f32,
    order:   StorageOrder,
    mean:    Tensor,
    rstd:    Tensor,
    scale:   Tensor,
    bias:    Tensor,

    phantom: PhantomData<T>,
}

input_tags!{
    InstanceNormOp {
        Input,
        Scale,
        Bias
    }
}

output_tags!{
    InstanceNormOp {
        Output,
        Mean,
        Rstd
    }
}

impl<T,Context> InstanceNormOp<T,Context> {
    
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
        const auto& beta = Input(BIAS);
        const int ndim = X.dim();
        const int64_t N = X.dim(0);
        const int64_t C = order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(ndim - 1);
        const int64_t HxW = X.numel() / (N * C);
        CAFFE_ENFORCE_EQ(gamma.numel(), C);
        CAFFE_ENFORCE_EQ(beta.numel(), C);
        auto* Y = Output(OUTPUT, X.sizes(), at::dtype<T>());
        const T* X_data = X.template data<T>();
        const T* gamma_data = gamma.template data<T>();
        const T* beta_data = beta.template data<T>();
        T* Y_data = Y->template mutable_data<T>();
        T* mean_data = nullptr;
        T* rstd_data = nullptr;
        if (OutputSize() >= 2) {
          auto* mean = Output(MEAN, {N, C}, at::dtype<T>());
          mean_data = mean->template mutable_data<T>();
        } else {
          ReinitializeTensor(
              &mean_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
          mean_data = mean_.template mutable_data<T>();
        }
        if (OutputSize() >= 3) {
          auto* rstd = Output(RSTD, {N, C}, at::dtype<T>());
          rstd_data = rstd->template mutable_data<T>();
        } else {
          ReinitializeTensor(
              &rstd_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
          rstd_data = rstd_.template mutable_data<T>();
        }
        switch (order_) {
          case StorageOrder::NCHW: {
            return RunOnDeviceWithOrderNCHW(
                N,
                C,
                HxW,
                X_data,
                gamma_data,
                beta_data,
                Y_data,
                mean_data,
                rstd_data);
          }
          case StorageOrder::NHWC: {
            return RunOnDeviceWithOrderNHWC(
                N,
                C,
                HxW,
                X_data,
                gamma_data,
                beta_data,
                Y_data,
                mean_data,
                rstd_data);
          }
          default: {
            CAFFE_THROW("Unknown storage order: ", order_);
          }
        }
        */
    }
}

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

register_cpu_operator!{
    InstanceNorm, 
    InstanceNormOp<f32, CPUContext>
}

allow_inplace!{InstanceNorm, vec![(0, 0)]}

num_inputs!{InstanceNorm, 3}

num_outputs!{InstanceNorm, (1,3)}

inputs!{InstanceNorm, 
    0 => ("input",            "The input 4-dimensional NCHW tensor to be operated on."),
    1 => ("scale",            "The input 1-dimensional scale tensor of size *C*."),
    2 => ("bias",             "The input 1-dimensional bias tensor of size *C*.")
}

outputs!{InstanceNorm, 
    0 => ("output",           "The output 4-dimensional tensor of the same shape as input."),
    1 => ("saved_mean",       "(Optional) Saved mean used during training to speed up gradient computation. Should not be used for testing."),
    2 => ("saved_inv_stdev",  "(Optional) Saved inverse stdev used during training to speed up gradient computation. Should not be used for testing.")
}

args!{InstanceNorm, 
    0 => ("epsilon",          "*(type: float; default: 1e-5)* The epsilon value to use to avoid division by zero."),
    1 => ("order",            "*(type: string; default: NCHW)* Specifies the order of the input data blob, where $N$ is batch size, $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is NHWC.")
}

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

#[inline] pub fn compute_fused_params<T>(
    n:       i64,
    c:       i64,
    mean:    *const T,
    rstd:    *const T,
    gamma:   *const T,
    beta:    *const T,
    scale:   *mut T,
    bias:    *mut T) 
{
    todo!();
    /*
        ConstEigenArrayMap<T> mean_arr(mean, C, N);
      ConstEigenArrayMap<T> rstd_arr(rstd, C, N);
      ConstEigenVectorArrayMap<T> gamma_arr(gamma, C);
      ConstEigenVectorArrayMap<T> beta_arr(beta, C);
      EigenArrayMap<T> scale_arr(scale, C, N);
      EigenArrayMap<T> bias_arr(bias, C, N);
      scale_arr = rstd_arr.colwise() * gamma_arr;
      bias_arr = (-scale_arr * mean_arr).colwise() + beta_arr;
    */
}

#[inline] pub fn instance_norm_forwardNHWC<T>(
    n:     i64,
    c:     i64,
    hxW:   i64,
    x:     *const T,
    scale: *const T,
    bias:  *const T,
    y:     *mut T) 
{
    todo!();
    /*
        ConstEigenArrayMap<T> scale_arr(scale, C, N);
      ConstEigenArrayMap<T> bias_arr(bias, C, N);
      for (int64_t i = 0; i < N; ++i) {
        ConstEigenArrayMap<T> X_arr(X + i * HxW * C, C, HxW);
        EigenArrayMap<T> Y_arr(Y + i * HxW * C, C, HxW);
        Y_arr = (X_arr.colwise() * scale_arr.col(i)).colwise() + bias_arr.col(i);
      }
    */
}

impl<T, Context> InstanceNormOp<T, Context> {
    
    #[inline] pub fn run_f32_on_cpu_device_with_order_nchw(
        &mut self, 
        n:      i64,
        c:      i64,
        hxW:    i64,
        x:      *const f32,
        gamma:  *const f32,
        beta:   *const f32,
        y:      *mut f32,
        mean:   *mut f32,
        rstd:   *mut f32) -> bool 
    {

        todo!();
        /*
            ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
      for (int64_t i = 0; i < N * C; ++i) {
        const float mean_val = X_arr.col(i).mean();
        float rstd_val =
            std::max(X_arr.col(i).square().mean() - mean_val * mean_val, 0.0f);
        rstd_val = 1.0f / std::sqrt(rstd_val + epsilon_);
        const int64_t c = i % C;
        const float scale = gamma[c] * rstd_val;
        const float bias = beta[c] - scale * mean_val;
        for (int64_t j = 0; j < HxW; ++j) {
          Y[i * HxW + j] = scale * X[i * HxW + j] + bias;
        }
        mean[i] = mean_val;
        rstd[i] = rstd_val;
      }
      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nhwc(
        &mut self, 
        n:       i64,
        c:       i64,
        hxW:     i64,
        x:       *const f32,
        gamma:   *const f32,
        beta:    *const f32,
        y:       *mut f32,
        mean:    *mut f32,
        rstd:    *mut f32) -> bool 
    {
        todo!();
        /*
            ReinitializeTensor(&scale_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&bias_, {N, C}, at::dtype<float>().device(CPU));
      float* scale_data = scale_.template mutable_data<float>();
      float* bias_data = bias_.template mutable_data<float>();
      const float c = 1.0f / static_cast<float>(HxW);
      EigenArrayMap<float> mean_arr(mean, C, N);
      EigenArrayMap<float> rstd_arr(rstd, C, N);
      for (int64_t n = 0; n < N; ++n) {
        ConstEigenArrayMap<float> X_arr(X + n * HxW * C, C, HxW);
        mean_arr.col(n) = X_arr.col(0);
        rstd_arr.col(n) = X_arr.col(0).square();
        for (int64_t i = 1; i < HxW; ++i) {
          mean_arr.col(n) += X_arr.col(i);
          rstd_arr.col(n) += X_arr.col(i).square();
        }
      }
      mean_arr *= c;
      rstd_arr = ((rstd_arr * c - mean_arr.square()).max(0.0f) + epsilon_).rsqrt();
      ComputeFusedParams<float>(
          N, C, mean, rstd, gamma, beta, scale_data, bias_data);
      InstanceNormForwardNHWC<float>(N, C, HxW, X, scale_data, bias_data, Y);
      return true;
        */
    }
}

#[inline] pub fn compute_internal_gradientsNHWC<T>(
    n:   i64,
    c:   i64,
    hxW: i64,
    dY:  *const T,
    x:   *const T,
    ds:  *mut T,
    db:  *mut T) 
{
    todo!();
    /*
        EigenArrayMap<T> ds_arr(ds, C, N);
      EigenArrayMap<T> db_arr(db, C, N);
      for (int64_t i = 0; i < N; ++i) {
        ConstEigenArrayMap<T> dY_arr(dY + i * C * HxW, C, HxW);
        ConstEigenArrayMap<T> X_arr(X + i * C * HxW, C, HxW);
        ds_arr.col(i) = dY_arr.col(0) * X_arr.col(0);
        db_arr.col(i) = dY_arr.col(0);
        for (int j = 1; j < HxW; ++j) {
          ds_arr.col(i) += dY_arr.col(j) * X_arr.col(j);
          db_arr.col(i) += dY_arr.col(j);
        }
      }
    */
}

#[inline] pub fn instance_norm_backwardNCHW<T>(
    n:      i64,
    c:      i64,
    hxW:    i64,
    dY:     *const T,
    x:      *const T,
    mean:   *const T,
    rstd:   *const T,
    gamma:  *const T,
    dX:     *mut T,
    ds:     *mut T,
    db:     *mut T) 
{
    todo!();
    /*
       const T scale = T(1) / static_cast<T>(HxW);
      ConstEigenArrayMap<T> dY_arr(dY, HxW, N * C);
      ConstEigenArrayMap<T> X_arr(X, HxW, N * C);
      for (int64_t i = 0; i < N * C; ++i) {
        const T ds_sum = (dY_arr.col(i) * X_arr.col(i)).sum();
        const T db_sum = dY_arr.col(i).sum();
        const int64_t c = i % C;
        const T c1 = rstd[i] * gamma[c];
        T c2 = ds_sum * gamma[c];
        T c3 = db_sum * gamma[c];
        c2 = (c3 * mean[i] - c2) * rstd[i] * rstd[i] * rstd[i] * scale;
        c3 = -c2 * mean[i] - c3 * rstd[i] * scale;
        for (int64_t j = 0; j < HxW; ++j) {
          const int64_t index = i * HxW + j;
          dX[index] = c1 * dY[index] + c2 * X[index] + c3;
        }
        ds[i] = ds_sum;
        db[i] = db_sum;
      }
    */
}


#[inline] pub fn instance_norm_backwardNHWC<T>(
    n:       i64,
    c:       i64,
    hxW:     i64,
    dY:      *const T,
    x:       *const T,
    ds:      *const T,
    db:      *const T,
    mean:    *const T,
    rstd:    *const T,
    gamma:   *const T,
    dX:      *mut T,
    c1:      *mut T,
    c2:      *mut T,
    c3:      *mut T) 
{
    todo!();
    /*
        const T scale = T(1) / static_cast<T>(HxW);
      ConstEigenArrayMap<T> ds_arr(ds, C, N);
      ConstEigenArrayMap<T> db_arr(db, C, N);
      ConstEigenArrayMap<T> mean_arr(mean, C, N);
      ConstEigenArrayMap<T> rstd_arr(rstd, C, N);
      ConstEigenVectorArrayMap<T> gamma_arr(gamma, C);
      EigenArrayMap<T> c1_arr(c1, C, N);
      EigenArrayMap<T> c2_arr(c2, C, N);
      EigenArrayMap<T> c3_arr(c3, C, N);
      c1_arr = rstd_arr.colwise() * gamma_arr;
      c2_arr = ds_arr.colwise() * gamma_arr;
      c3_arr = db_arr.colwise() * gamma_arr;
      c2_arr = (c3_arr * mean_arr - c2_arr) * rstd_arr.cube() * scale;
      c3_arr = -c2_arr * mean_arr - c3_arr * rstd_arr * scale;
      for (int64_t i = 0; i < N; ++i) {
        ConstEigenArrayMap<T> dY_arr(dY + i * HxW * C, C, HxW);
        ConstEigenArrayMap<T> X_arr(X + i * HxW * C, C, HxW);
        EigenArrayMap<T> dX_arr(dX + i * HxW * C, C, HxW);
        dX_arr =
            (dY_arr.colwise() * c1_arr.col(i) + X_arr.colwise() * c2_arr.col(i))
                .colwise() +
            c3_arr.col(i);
      }
    */
}

#[inline] pub fn gamma_beta_backward<T>(
    n:      i64,
    c:      i64,
    ds:     *const T,
    db:     *const T,
    mean:   *const T,
    rstd:   *const T,
    dgamma: *mut T,
    dbeta:  *mut T) 
{
    todo!();
    /*
        ConstEigenArrayMap<T> ds_arr(ds, C, N);
      ConstEigenArrayMap<T> db_arr(db, C, N);
      ConstEigenArrayMap<T> mean_arr(mean, C, N);
      ConstEigenArrayMap<T> rstd_arr(rstd, C, N);
      EigenVectorArrayMap<T> dgamma_arr(dgamma, C);
      EigenVectorArrayMap<T> dbeta_arr(dbeta, C);
      dgamma_arr =
          (ds_arr.col(0) - db_arr.col(0) * mean_arr.col(0)) * rstd_arr.col(0);
      dbeta_arr = db_arr.col(0);
      for (int64_t i = 1; i < N; ++i) {
        dgamma_arr +=
            (ds_arr.col(i) - db_arr.col(i) * mean_arr.col(i)) * rstd_arr.col(i);
        dbeta_arr += db_arr.col(i);
      }
    */
}

impl InstanceNormGradientOp<f32, CPUContext> {

    #[inline] pub fn compute_moments(
        n:    i64,
        c:    i64,
        hxW:  i64,
        x:    *const f32,
        mean: *mut f32,
        rstd: *mut f32) 
    {
        todo!();
        /*
            if (order_ == StorageOrder::NCHW) {
            const std::array<int, 2> X_dims = {static_cast<int>(N * C),
                                               static_cast<int>(HxW)};
            const std::array<int, 2> Y_dims = {static_cast<int>(N * C), 1};
            math::Moments<float, CPUContext>(
                2, X_dims.data(), Y_dims.data(), X, mean, rstd, &context_);
            math::InvStd<float, CPUContext>(N * C, epsilon_, rstd, rstd, &context_);
          } else {
            const float c = 1.0f / static_cast<float>(HxW);
            EigenArrayMap<float> mean_arr(mean, C, N);
            EigenArrayMap<float> rstd_arr(rstd, C, N);
            for (int64_t i = 0; i < N; ++i) {
              ConstEigenArrayMap<float> X_arr(X + i * HxW * C, C, HxW);
              mean_arr.col(i) = X_arr.col(0);
              rstd_arr.col(i) = X_arr.col(0).square();
              for (int64_t j = 1; j < HxW; ++j) {
                mean_arr.col(i) += X_arr.col(j);
                rstd_arr.col(i) += X_arr.col(j).square();
              }
            }
            mean_arr *= c;
            rstd_arr =
                ((rstd_arr * c - mean_arr.square()).max(0.0f) + epsilon_).rsqrt();
          }
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nchw(
        &mut self, 
        n:      i64,
        c:      i64,
        hxW:    i64,
        dY:     *const f32,
        x:      *const f32,
        mean:   *const f32,
        rstd:   *const f32,
        gamma:  *const f32,
        dX:     *mut f32,
        dgamma: *mut f32,
        dbeta:  *mut f32) -> bool 
    {
        todo!();
        /*
            ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CPU));
      float* ds_data = ds_.mutable_data<float>();
      float* db_data = db_.mutable_data<float>();
      InstanceNormBackwardNCHW<float>(
          N, C, HxW, dY, X, mean, rstd, gamma, dX, ds_data, db_data);
      GammaBetaBackward<float>(N, C, ds_data, db_data, mean, rstd, dgamma, dbeta);
      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nhwc(
        &mut self, 
        n:       i64,
        c:       i64,
        hxW:     i64,
        dY:      *const f32,
        x:       *const f32,
        mean:    *const f32,
        rstd:    *const f32,
        gamma:   *const f32,
        dX:      *mut f32,
        dgamma:  *mut f32,
        dbeta:   *mut f32) -> bool 
    {
        todo!();
        /*
            ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CPU));
      float* ds_data = ds_.mutable_data<float>();
      float* db_data = db_.mutable_data<float>();
      ComputeInternalGradientsNHWC<float>(N, C, HxW, dY, X, ds_data, db_data);
      ReinitializeTensor(&c1_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&c2_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&c3_, {N, C}, at::dtype<float>().device(CPU));
      float* c1_data = c1_.mutable_data<float>();
      float* c2_data = c2_.mutable_data<float>();
      float* c3_data = c3_.mutable_data<float>();
      InstanceNormBackwardNHWC<float>(
          N,
          C,
          HxW,
          dY,
          X,
          ds_data,
          db_data,
          mean,
          rstd,
          gamma,
          dX,
          c1_data,
          c2_data,
          c3_data);
      GammaBetaBackward<float>(N, C, ds_data, db_data, mean, rstd, dgamma, dbeta);
      return true;
        */
    }
}

register_cpu_operator!{
    InstanceNormGradient,
    InstanceNormGradientOp<f32, CPUContext>
}

num_inputs!{InstanceNormGradient, (4,6)}

num_outputs!{InstanceNormGradient, 3}

register_gradient!{InstanceNorm, GetInstanceNormGradient}

pub struct GetInstanceNormGradient { }

impl GetGradientDefs for GetInstanceNormGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            std::vector<std::string> inputs = {I(0), I(1), I(2), GO(0)};
        if (def_.output_size() >= 2) {
          inputs.push_back(O(1));
        }
        if (def_.output_size() >= 3) {
          inputs.push_back(O(2));
        }
        return SingleGradientDef(
            "InstanceNormGradient",
            "",
            inputs,
            std::vector<std::string>({GI(0), GI(1), GI(2)}));
        */
    }
}
