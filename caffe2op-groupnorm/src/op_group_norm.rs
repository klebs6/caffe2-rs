crate::ix!();

/**
  | Group Normalization (GN) operation:
  | https://arxiv.org/abs/1803.08494
  |
  */
pub struct GroupNormOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:  OperatorStorage,
    context:  Context,

    group:    i32,
    epsilon:  f32,
    order:    StorageOrder,
    is_test:  bool,
    mu:       Tensor,
    rsig:     Tensor,
    scale:    Tensor,
    bias:     Tensor,

    /**
      | Input: X, gamma, beta
      | 
      | Output: Y, mu, inv_sig
      |
      */
    phantom:  PhantomData<T>,
}

/**
  | Input: X, gamma, beta;
  | 
  | Output: Y, mu, sig
  |
  */
num_inputs!{GroupNorm, 3}

num_outputs!{GroupNorm, (1,3)}

inputs!{GroupNorm, 
    0 => ("X",      ">=4D feature map input of shape (N, C, H, W) or (N, C, T, H, W)"),
    1 => ("gamma",  "The scale as a 1-dimensional tensor of size C to be applied to the output."),
    2 => ("beta",   "The bias as a 1-dimensional tensor of size C to be applied to the output.")
}

outputs!{GroupNorm, 
    0 => ("Y",      "The output >=4-dimensional tensor of the same shape as X."),
    1 => ("mean",   "The mean of shape (N, G). For backward usage or reference. Cannot be used as activations."),
    2 => ("std",    "The std of shape (N, G). For backward usage or reference. Cannot be used as activations.")
}

args!{GroupNorm, 
    0 => ("num_groups", "(int) default 32; number of groups used by GN."),
    1 => ("epsilon",    "(float) default 1e-5; small constant added to var.")
}

input_tags!{
    GroupNormOp {
        Input,
        Gamma,
        Beta
    }
}

output_tags!{
    GroupNormOp {
        Output,
        Mu,
        InvSigma
    }
}

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

///----------------------------------------
pub struct GroupNormGradientOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    group:    i32,
    order:    StorageOrder,
    ds:       Tensor,
    db:       Tensor,
    dY_scale: Tensor,
    x_scale:  Tensor,
    bias:     Tensor,
    ones:     Tensor,

    /**
      | Input: dY, X, gamma, beta, mu, inv_sig
      | 
      | Output: dX, dgamma, dbeta
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{GroupNormGradient, 6}

num_outputs!{GroupNormGradient, 3}

input_tags!{
    GroupNormGradientOp {
        OutputGrad,
        Input,
        Gamma,
        Beta,
        Mu,
        InvSigma
    }
}

output_tags!{
    GroupNormGradientOp {
        InputGrad,
        GammaGrad,
        BetaGrad
    }
}

impl<T, Context> GroupNormGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "group", group_, 32),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<std::string>("order", "NCHW"))) 

        CAFFE_ENFORCE_NE(
            order_,
            StorageOrder::UNKNOWN,
            "order should be either \"NCHW\" or \"NHWC\".");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dY = Input(OUTPUT_GRAD);
        const auto& X = Input(INPUT);
        const auto& gamma = Input(GAMMA);
        const auto& beta = Input(BETA);
        const auto& mu = Input(MU);
        const auto& rsig = Input(INV_SIGMA);
        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
        const int HxW = X.numel() / (N * C);
        CAFFE_ENFORCE_EQ(C % group_, 0);
        CAFFE_ENFORCE_EQ(gamma.numel(), C);
        CAFFE_ENFORCE_EQ(beta.numel(), C);
        const int G = group_;
        const int K = C / G;
        auto* dX = Output(INPUT_GRAD, X.sizes(), at::dtype<T>());
        auto* dgamma = Output(GAMMA_GRAD, gamma.sizes(), at::dtype<T>());
        auto* dbeta = Output(BETA_GRAD, beta.sizes(), at::dtype<T>());
        if (order_ == StorageOrder::NCHW) {
          return RunOnDeviceWithOrderNCHW(
              N,
              G,
              K,
              HxW,
              dY.template data<T>(),
              X.template data<T>(),
              mu.template data<T>(),
              rsig.template data<T>(),
              gamma.template data<T>(),
              dX->template mutable_data<T>(),
              dgamma->template mutable_data<T>(),
              dbeta->template mutable_data<T>());
        } else {
          return RunOnDeviceWithOrderNHWC(
              N,
              G,
              K,
              HxW,
              dY.template data<T>(),
              X.template data<T>(),
              mu.template data<T>(),
              rsig.template data<T>(),
              gamma.template data<T>(),
              dX->template mutable_data<T>(),
              dgamma->template mutable_data<T>(),
              dbeta->template mutable_data<T>());
        }
        */
    }
}

/**
  | GroupNorm op in Caffe2 for CPU
  | Written by Kaiming He
  | Improved by Xiaomeng Yang
  | Somewhat degraded and pounded mercilessly by klebz
  | see https://arxiv.org/abs/1803.08494
  | This is a stand-alone op: Y = gamma * (X - mu) / sig + beta
  */

/**
  | Math:
  | Y = gamma * (X - mu) * rsig + beta
  | let s = gamma * rsig
  | let b = beta - gamma * mu * rsig
  | Y = s * X + b
  | let n = K * HxW
  | dL/dX = dL/dY * dY/dX = dL/dY * (d(s * X)/dX + db/dX)
  | d(s * X)/dX = s + X * ds/dX = s + gamma * X * drsig/dX
  | db/dX = -gamma * u * drsig/dX - gamma * rsig * dmu/dX
  | drsig/dX = -rsig^3 * (X - mu) / n
  | dmu/dX = 1 / n
  */
pub fn compute_internal_gradients<T, StorageOrder>(
    N: i32,
    C: i32,
    HxW: i32,
    dY: *const T,
    X:  *const T,
    ds: *mut T,
    db: *mut T) 
{
    todo!("dispatch"); 
    /* */ 
}

#[inline] pub fn compute_internal_gradients_f32_nchw(
    n:    i32,
    c:    i32,
    hxW:  i32,
    dY:   *const f32,
    x:    *const f32,
    ds:   *mut f32,
    db:   *mut f32)  
{
    todo!();
    /*
        ConstEigenArrayMap<float> dY_arr(dY, HxW, N * C);
      ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
      for (int i = 0; i < N * C; ++i) {
        ds[i] = (dY_arr.col(i) * X_arr.col(i)).sum();
        db[i] = dY_arr.col(i).sum();
      }
    */
}

#[inline] pub fn compute_internal_gradients_f32_nhwc(
    n:   i32,
    c:   i32,
    hxW: i32,
    dY:  *const f32,
    x:   *const f32,
    ds:  *mut f32,
    db:  *mut f32)  
{
    todo!();
    /*
        EigenArrayMap<float> ds_arr(ds, C, N);
      EigenArrayMap<float> db_arr(db, C, N);
      for (int i = 0; i < N; ++i) {
        ConstEigenArrayMap<float> dY_arr(dY + i * C * HxW, C, HxW);
        ConstEigenArrayMap<float> X_arr(X + i * C * HxW, C, HxW);
        ds_arr.col(i) = dY_arr.col(0) * X_arr.col(0);
        db_arr.col(i) = dY_arr.col(0);
        for (int j = 1; j < HxW; ++j) {
          ds_arr.col(i) += dY_arr.col(j) * X_arr.col(j);
          db_arr.col(i) += dY_arr.col(j);
        }
      }
    */
}

#[inline] pub fn compute_gradient_fused_params<T>(
    n:          i32,
    g:          i32,
    k:          i32,
    hxW:        i32,
    ds:         *const T,
    db:         *const T,
    mu:         *const T,
    rsig:       *const T,
    gamma:      *const T,
    dY_scale:   *mut T,
    x_scale:    *mut T,
    bias:       *mut T) 
{
    todo!();
    /*
        ConstEigenArrayMap<T> rsig_arr(rsig, G, N);
      ConstEigenArrayMap<T> gamma_arr(gamma, K, G);
      for (int i = 0; i < N; ++i) {
        EigenArrayMap<T>(dY_scale + i * G * K, K, G) =
            gamma_arr.rowwise() * (rsig_arr.col(i).transpose());
      }
      ConstEigenVectorArrayMap<T> mu_arr(mu, N * G);
      ConstEigenVectorArrayMap<T> rsig_vec(rsig, N * G);
      EigenVectorArrayMap<T> X_scale_arr(X_scale, N * G);
      EigenVectorArrayMap<T> bias_arr(bias, N * G);
      for (int i = 0; i < N; ++i) {
        ConstEigenArrayMap<T> ds_arr(ds + i * G * K, K, G);
        ConstEigenArrayMap<T> db_arr(db + i * G * K, K, G);
        for (int j = 0; j < G; ++j) {
          X_scale_arr(i * G + j) = (ds_arr.col(j) * gamma_arr.col(j)).sum();
          bias_arr(i * G + j) = (db_arr.col(j) * gamma_arr.col(j)).sum();
        }
      }
      const T alpha = T(1) / static_cast<T>(K * HxW);
      X_scale_arr = (bias_arr * mu_arr - X_scale_arr) * rsig_vec.cube() * alpha;
      bias_arr = -X_scale_arr * mu_arr - bias_arr * rsig_vec * alpha;
    */
}

#[inline] pub fn group_norm_backward_i32_nchw(
    n:         i32,
    g:         i32,
    k:         i32,
    hxW:       i32,
    dY_scale:  *const f32,
    dY:        *const f32,
    x_scale:   *const f32,
    x:         *const f32,
    bias:      *const f32,
    dX:        *mut f32)  
{
    todo!();
    /*
        const int C = G * K;
      ConstEigenArrayMap<float> dY_arr(dY, HxW, N * C);
      ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
      EigenArrayMap<float> dX_arr(dX, HxW, N * C);
      for (int i = 0; i < N * G; ++i) {
        for (int j = 0; j < K; ++j) {
          const int c = i * K + j;
          dX_arr.col(c) =
              dY_arr.col(c) * dY_scale[c] + X_arr.col(c) * X_scale[i] + bias[i];
        }
      }
    */
}

#[inline] pub fn group_norm_backward_f32_nhwc(
    n:          i32,
    g:          i32,
    k:          i32,
    hxW:        i32,
    dY_scale:   *const f32,
    dY:         *const f32,
    x_scale:    *const f32,
    x:          *const f32,
    bias:       *const f32,
    dX:         *mut f32)
{
    todo!();
    /*
        const int C = G * K;
      ConstEigenArrayMap<float> X_scale_arr(X_scale, G, N);
      ConstEigenArrayMap<float> bias_arr(bias, G, N);
      for (int n = 0; n < N; ++n) {
        ConstEigenArrayMap<float> dY_scale_arr(dY_scale + n * C, K, G);
        for (int i = 0; i < HxW; ++i) {
          const int m = n * HxW + i;
          ConstEigenArrayMap<float> dY_arr(dY + m * C, K, G);
          ConstEigenArrayMap<float> X_arr(X + m * C, K, G);
          EigenArrayMap<float> dX_arr(dX + m * C, K, G);
          dX_arr = (dY_arr * dY_scale_arr +
                    X_arr.rowwise() * X_scale_arr.col(n).transpose())
                       .rowwise() +
              bias_arr.col(n).transpose();
        }
      }
    */
}

#[inline] pub fn gamma_beta_backward<T>(
    n:        i32,
    g:        i32,
    k:        i32,
    ds:       *const T,
    db:       *const T,
    mu:       *const T,
    rsig:     *const T,
    dgamma:   *mut T,
    dbeta:    *mut T) 
{
    todo!();
    /*
        const int C = G * K;
      ConstEigenArrayMap<T> ds0_arr(ds, K, G);
      ConstEigenArrayMap<T> db0_arr(db, K, G);
      ConstEigenArrayMap<T> mu_arr(mu, G, N);
      ConstEigenArrayMap<T> rsig_arr(rsig, G, N);
      EigenArrayMap<T> dgamma_arr(dgamma, K, G);
      EigenArrayMap<T> dbeta_arr(dbeta, K, G);
      dgamma_arr =
          (ds0_arr - db0_arr.rowwise() * mu_arr.col(0).transpose()).rowwise() *
          rsig_arr.col(0).transpose();
      dbeta_arr = db0_arr;
      for (int i = 1; i < N; ++i) {
        ConstEigenArrayMap<T> dsi_arr(ds + i * C, K, G);
        ConstEigenArrayMap<T> dbi_arr(db + i * C, K, G);
        dgamma_arr +=
            (dsi_arr - dbi_arr.rowwise() * mu_arr.col(i).transpose()).rowwise() *
            rsig_arr.col(i).transpose();
        dbeta_arr += dbi_arr;
      }
    */
}

impl GroupNormOp<f32, CPUContext> {

    #[inline] pub fn compute_fused_params(
        &mut self, 
        n:     i32,
        g:     i32,
        k:     i32,
        mu:    *const f32,
        rsig:  *const f32,
        gamma: *const f32,
        beta:  *const f32,
        scale: *mut f32,
        bias:  *mut f32)  
    {
        todo!();
        /*
            const int C = G * K;
      ConstEigenArrayMap<float> mu_arr(mu, G, N);
      ConstEigenArrayMap<float> rsig_arr(rsig, G, N);
      ConstEigenArrayMap<float> gamma_arr(gamma, K, G);
      ConstEigenArrayMap<float> beta_arr(beta, K, G);
      for (int i = 0; i < N; ++i) {
        EigenArrayMap<float> scale_arr(scale + i * C, K, G);
        EigenArrayMap<float> bias_arr(bias + i * C, K, G);
        scale_arr = gamma_arr.rowwise() * rsig_arr.col(i).transpose();
        bias_arr = beta_arr - scale_arr.rowwise() * mu_arr.col(i).transpose();
      }
        */
    }
    
    #[inline] pub fn group_norm_forwardNCHW(
        &mut self, 
        n:      i32,
        c:      i32,
        hxW:    i32,
        x:      *const f32,
        scale:  *const f32,
        bias:   *const f32,
        y:      *mut f32)  
    {
        todo!();
        /*
            EigenArrayMap<float>(Y, HxW, N * C) =
          (ConstEigenArrayMap<float>(X, HxW, N * C).rowwise() *
           ConstEigenVectorArrayMap<float>(scale, N * C).transpose())
              .rowwise() +
          ConstEigenVectorArrayMap<float>(bias, N * C).transpose();
        */
    }
    
    #[inline] pub fn group_norm_forwardNHWC(
        &mut self, 
        n:      i32,
        c:      i32,
        hxW:    i32,
        x:      *const f32,
        scale:  *const f32,
        bias:   *const f32,
        y:      *mut f32)  
    {
        todo!();
        /*
            const int stride = HxW * C;
      for (int i = 0; i < N; ++i) {
        EigenArrayMap<float>(Y + i * stride, C, HxW) =
            (ConstEigenArrayMap<float>(X + i * stride, C, HxW).colwise() *
             ConstEigenVectorArrayMap<float>(scale + i * C, C))
                .colwise() +
            ConstEigenVectorArrayMap<float>(bias + i * C, C);
      }
        */
    }
    
    #[inline] pub fn run_f32_on_cpu_device_with_orderNHWC(
        &mut self, 
        n:      i32,
        g:      i32,
        k:      i32,
        hxW:    i32,
        x:      *const f32,
        gamma:  *const f32,
        beta:   *const f32,
        y:      *mut f32,
        mu:     *mut f32,
        rsig:   *mut f32) -> bool 
    {
        todo!();
        /*
            const int C = G * K;
      ReinitializeTensor(&scale_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&bias_, {N, C}, at::dtype<float>().device(CPU));
      float* scale_data = scale_.mutable_data<float>();
      float* bias_data = bias_.mutable_data<float>();
      EigenVectorArrayMap<float> mu_arr(mu, N * G);
      EigenVectorArrayMap<float> rsig_arr(rsig, N * G);
      mu_arr.setZero();
      rsig_arr.setZero();
      for (int n = 0; n < N; ++n) {
        for (int i = 0; i < HxW; ++i) {
          const int m = n * HxW + i;
          ConstEigenArrayMap<float> X_arr(X + m * C, K, G);
          for (int j = 0; j < G; ++j) {
            mu_arr(n * G + j) += X_arr.col(j).sum();
            rsig_arr(n * G + j) += X_arr.col(j).square().sum();
          }
        }
      }
      const float scale = 1.0f / static_cast<float>(K * HxW);
      mu_arr *= scale;
      rsig_arr = (rsig_arr * scale - mu_arr.square() + epsilon_).rsqrt();
      ComputeFusedParams(N, G, K, mu, rsig, gamma, beta, scale_data, bias_data);
      GroupNormForwardNHWC(N, C, HxW, X, scale_data, bias_data, Y);
      return true;
        */
    }
}

impl GroupNormGradientOp<f32, CPUContext> {

    /**
      | Math:
      | let: s = gamma * rsig
      | let: b = beta - mu * gamma * rsig
      | then: Y = s * X + b
      */
    #[inline] pub fn run_on_device_with_orderNCHW(
        &mut self, 
        n:             i32,
        g:             i32,
        k:             i32,
        hxW:           i32,
        dY_data:       *const f32,
        x_data:        *const f32,
        mu_data:       *const f32,
        rsig_data:     *const f32,
        gamma_data:    *const f32,
        dX_data:       *mut f32,
        dgamma_data:   *mut f32,
        dbeta_data:    *mut f32) -> bool 
    {
        todo!();
        /*
            const int C = G * K;
      ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&dY_scale_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&X_scale_, {N, G}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&bias_, {N, G}, at::dtype<float>().device(CPU));
      float* ds_data = ds_.mutable_data<float>();
      float* db_data = db_.mutable_data<float>();
      float* dY_scale_data = dY_scale_.mutable_data<float>();
      float* X_scale_data = X_scale_.mutable_data<float>();
      float* bias_data = bias_.mutable_data<float>();
      ComputeInternalGradients<float, StorageOrder::NCHW>(
          N, C, HxW, dY_data, X_data, ds_data, db_data);
      ComputeGradientFusedParams<float>(
          N,
          G,
          K,
          HxW,
          ds_data,
          db_data,
          mu_data,
          rsig_data,
          gamma_data,
          dY_scale_data,
          X_scale_data,
          bias_data);
      GroupNormBackward<float, StorageOrder::NCHW>(
          N,
          G,
          K,
          HxW,
          dY_scale_data,
          dY_data,
          X_scale_data,
          X_data,
          bias_data,
          dX_data);
      GammaBetaBackward<float>(
          N, G, K, ds_data, db_data, mu_data, rsig_data, dgamma_data, dbeta_data);
      return true;
        */
    }

    #[inline] pub fn run_on_device_with_order_nhwc<T, Context>(
        &mut self,
        n:            i32,
        g:            i32,
        k:            i32,
        hxW:          i32,
        dY_data:      *const T,
        x_data:       *const T,
        mu_data:      *const T,
        rsig_data:    *const T,
        gamma_data:   *const T,
        dX_data:      *mut T,
        dgamma_data:  *mut T,
        dbeta_data:   *mut T) -> bool 
    {
        todo!();
        /*
            const int C = G * K;
          ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CPU));
          ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CPU));
          ReinitializeTensor(&dY_scale_, {N, C}, at::dtype<float>().device(CPU));
          ReinitializeTensor(&X_scale_, {N, G}, at::dtype<float>().device(CPU));
          ReinitializeTensor(&bias_, {N, G}, at::dtype<float>().device(CPU));
          float* ds_data = ds_.mutable_data<float>();
          float* db_data = db_.mutable_data<float>();
          float* dY_scale_data = dY_scale_.mutable_data<float>();
          float* X_scale_data = X_scale_.mutable_data<float>();
          float* bias_data = bias_.mutable_data<float>();
          ComputeInternalGradients<float, StorageOrder::NHWC>(
              N, C, HxW, dY_data, X_data, ds_data, db_data);
          ComputeGradientFusedParams<float>(
              N,
              G,
              K,
              HxW,
              ds_data,
              db_data,
              mu_data,
              rsig_data,
              gamma_data,
              dY_scale_data,
              X_scale_data,
              bias_data);
          GroupNormBackward<float, StorageOrder::NHWC>(
              N,
              G,
              K,
              HxW,
              dY_scale_data,
              dY_data,
              X_scale_data,
              X_data,
              bias_data,
              dX_data);
          GammaBetaBackward<float>(
              N, G, K, ds_data, db_data, mu_data, rsig_data, dgamma_data, dbeta_data);
          return true;
        */
    }
}

register_cpu_operator!{GroupNorm, GroupNormOp<f32, CPUContext>}

register_cpu_operator!{
    GroupNormGradient,
    GroupNormGradientOp<f32, CPUContext>
}

/**
  | Warning: mu and rsig are for backward usage or
  | reference. They should NOT be used as forward
  | activations as they have no direct gradients
  | computed.
  */
pub struct GetGroupNormGradient;

impl GetGradientDefs for GetGroupNormGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "GroupNormGradient",
            "",
            std::vector<std::string>{GO(0), I(0), I(1), I(2), O(1), O(2)},
            std::vector<std::string>{GI(0), GI(1), GI(2)});
        */
    }
}

register_gradient!{GroupNorm, GetGroupNormGradient}
