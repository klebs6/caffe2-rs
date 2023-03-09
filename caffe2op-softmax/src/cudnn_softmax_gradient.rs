crate::ix!();

pub struct CudnnSoftmaxGradientOp {
    storage:        OperatorStorage,
    context:        CUDAContext,
    cudnn_wrapper:  CudnnWrapper,
    axis:           i32,
    desc:           CudnnTensorDescriptor,
    dims:           Vec<i64>,
}

register_cudnn_operator!{Softmax, CudnnSoftmaxOp}

register_cudnn_operator!{SoftmaxGradient, CudnnSoftmaxGradientOp}

impl Drop for CudnnSoftmaxGradientOp {

    fn drop(&mut self) {
        todo!();
        /* 
        CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(desc_));
       */
    }
}

impl CudnnSoftmaxGradientOp {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<CUDAContext>(std::forward<Args>(args)...),
            cudnn_wrapper_(&context_),
            axis_(OperatorStorage::GetSingleArgument<int>("axis", 1)) 

        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&desc_));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            auto& Y = Input(0);
        auto& dY = Input(1);

        const auto canonical_axis = Y.canonical_axis_index(axis_);
        const int N = Y.size_to_dim(canonical_axis);
        const int D = Y.size_from_dim(canonical_axis);

        CHECK_EQ(Y.sizes(), dY.sizes());
        auto* dX = Output(0, Y.sizes(), at::dtype<T>());
        auto* dX_data = dX->template mutable_data<T>();
        if (N == 0 || D == 0) {
          return true;
        }
        if (dims_ != Y.sizes()) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
              desc_,
              GetCudnnTensorFormat(StorageOrder::NCHW),
              cudnnTypeWrapper<T>::type,
              N,
              D,
              1,
              1));
          dims_ = Y.sizes().vec();
        }
        CUDNN_ENFORCE(cudnnSoftmaxBackward(
            cudnn_wrapper_.inline_cudnn_handle(),
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            cudnnTypeWrapper<T>::kOne(),
            desc_,
            Y.template data<T>(),
            desc_,
            dY.template data<T>(),
            cudnnTypeWrapper<T>::kZero(),
            desc_,
            dX_data));
        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
        */
    }
}
