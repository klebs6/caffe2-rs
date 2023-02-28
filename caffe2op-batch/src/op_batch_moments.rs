crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    CPUContext,
    StorageOrder,
    OperatorDef
};

pub struct BatchMomentsOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    order:   StorageOrder,
    phantom: PhantomData<T>,
}

num_inputs!{BatchMoments, 1}

num_outputs!{BatchMoments, 2}

register_cpu_operator!{BatchMoments, BatchMomentsOp<f32, CPUContext>}

impl<T,Context> BatchMomentsOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
                 order_(StringToStorageOrder(
                         this->template GetSingleArgument<std::string>("order", "NCHW"))) 
                     CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
        const int HxW = X.numel() / (N * C);
        auto* mu = Output(0, {C}, at::dtype<T>());
        auto* var = Output(1, {C}, at::dtype<T>());
        const T* X_data = X.template data<T>();
        T* mu_data = mu->template mutable_data<T>();
        T* var_data = var->template mutable_data<T>();
        return order_ == StorageOrder::NCHW
            ? ComputeBatchMomentsNCHW(N, C, HxW, X_data, mu_data, var_data)
            : ComputeBatchMomentsNHWC(N, C, HxW, X_data, mu_data, var_data);
        */
    }
}

impl BatchMomentsOp<f32, CPUContext> {

    #[inline] pub fn compute_batch_moments_nchw(
        &mut self, 
        n:      i32,
        c:      i32,
        hxW:    i32,
        x:      *const f32,
        mu:     *mut f32,
        var:    *mut f32) -> bool 
    {
        todo!();
        /*
            math::Set<float, CPUContext>(C, 0.0f, mu, &context_);
      math::Set<float, CPUContext>(C, 0.0f, var, &context_);
      EigenVectorArrayMap<float> mu_arr(mu, C);
      EigenVectorArrayMap<float> var_arr(var, C);
      const float* X_ptr = X;
      const int stride = C * HxW;
      for (int i = 0; i < N; ++i) {
        ConstEigenArrayMap<float> X_arr(X_ptr, HxW, C);
        mu_arr += X_arr.colwise().sum();
        var_arr += X_arr.square().colwise().sum();
        X_ptr += stride;
      }
      const float scale = 1.0f / static_cast<float>(N * HxW);
      math::Scale<float, float, CPUContext>(C, scale, mu, mu, &context_);
      math::Scale<float, float, CPUContext>(C, scale, var, var, &context_);
      return true;
        */
    }
    
    #[inline] pub fn compute_batch_moments_nhwc(
        &mut self, 
        n:    i32,
        c:    i32,
        hxW:  i32,
        x:    *const f32,
        mu:   *mut f32,
        var:  *mut f32) -> bool 
    {
        todo!();
        /*
            ConstEigenArrayMap<float> X_arr(X, C, N * HxW);
      EigenVectorMap<float>(mu, C) = X_arr.rowwise().mean();
      EigenVectorMap<float>(var, C) = X_arr.square().rowwise().mean();
      return true;
        */
    }
}

pub struct BatchMomentsGradientOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;

    storage: OperatorStorage,
    context: Context,

    order:   StorageOrder,
    phantom: PhantomData<T>,
}

num_inputs!{BatchMomentsGradient, 3}

num_outputs!{BatchMomentsGradient, 1}

register_cpu_operator!{BatchMomentsGradient, BatchMomentsGradientOp<f32, CPUContext>}

impl<T, Context> BatchMomentsGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<std::string>("order", "NCHW"))) 

        CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dmu = Input(0);
        const auto& dvar = Input(1);
        const auto& X = Input(2);

        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
        const int HxW = X.numel() / (N * C);
        auto* dX = Output(0, X.sizes(), at::dtype<T>());
        const T* dmu_data = dmu.template data<T>();
        const T* dvar_data = dvar.template data<T>();
        const T* X_data = X.template data<T>();
        T* dX_data = dX->template mutable_data<T>();
        return order_ == StorageOrder::NCHW
            ? ComputeBatchMomentsGradientNCHW(
                  N, C, HxW, dmu_data, dvar_data, X_data, dX_data)
            : ComputeBatchMomentsGradientNHWC(
                  N, C, HxW, dmu_data, dvar_data, X_data, dX_data);
        */
    }
}

impl BatchMomentsGradientOp<f32, CPUContext> {

    #[inline] pub fn compute_batch_moments_gradientNCHW(
        &mut self, 
        n:     i32,
        c:     i32,
        hxW:   i32,
        dmu:   *const f32,
        dvar:  *const f32,
        x:     *const f32,
        dX:    *mut f32) -> bool 
    {

        todo!();
        /*
            ConstEigenVectorArrayMap<float> dmu_arr(dmu, C);
      ConstEigenVectorArrayMap<float> dvar_arr(dvar, C);
      const float* X_ptr = X;
      float* dX_ptr = dX;
      const int stride = C * HxW;
      for (int i = 0; i < N; ++i) {
        EigenArrayMap<float> dX_arr(dX_ptr, HxW, C);
        dX_arr = ConstEigenArrayMap<float>(X_ptr, HxW, C).rowwise() *
            dvar_arr.transpose() * 2.0f;
        dX_arr.rowwise() += dmu_arr.transpose();
        X_ptr += stride;
        dX_ptr += stride;
      }
      const float scale = 1.0f / static_cast<float>(N * HxW);
      math::Scale<float, float, CPUContext>(N * C * HxW, scale, dX, dX, &context_);
      return true;
        */
    }
    
    #[inline] pub fn compute_batch_moments_gradientNHWC(
        &mut self, 
        n:     i32,
        c:     i32,
        hxW:   i32,
        dmu:   *const f32,
        dvar:  *const f32,
        x:     *const f32,
        dX:    *mut f32) -> bool 
    {
        todo!();
        /*
            const float scale = 1.0f / static_cast<float>(N * HxW);
      EigenArrayMap<float> dX_arr(dX, C, N * HxW);
      dX_arr = ConstEigenArrayMap<float>(X, C, N * HxW).colwise() *
          ConstEigenVectorArrayMap<float>(dvar, C) * 2.0f;
      dX_arr.colwise() += ConstEigenVectorArrayMap<float>(dmu, C);
      math::Scale<float, float, CPUContext>(N * C * HxW, scale, dX, dX, &context_);
      return true;
        */
    }
}

pub struct GetBatchMomentsGradient;

impl GetGradientDefs for GetBatchMomentsGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BatchMomentsGradient",
            "",
            std::vector<std::string>{GO(0), GO(1), I(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{BatchMoments, GetBatchMomentsGradient}
