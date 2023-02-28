crate::ix!();

use crate::{
    CudnnTensorDescriptor,
    CudnnActivationDescriptor,
    CudnnWrapper,
    OperatorStorage,
    CudnnDataType,
    CUDAContext,
};

pub struct CudnnActivationOpBase {

    //USE_OPERATOR_FUNCTIONS(CUDAContext);
    storage: OperatorStorage,
    context: CUDAContext,

    cudnn_wrapper: CudnnWrapper,
    data_desc:     CudnnTensorDescriptor,
    act_desc:      CudnnActivationDescriptor,
    input_size:    i32,
}

impl CudnnActivationOpBase {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CUDAContext>(std::forward<Args>(args)...),
            cudnn_wrapper_(&context_) 

        CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));
        CUDNN_ENFORCE(cudnnCreateActivationDescriptor(&act_desc_));
        */
    }

    #[inline] pub fn set_tensor_descriptor(
        &mut self, 
        data_type: CudnnDataType,
        data_size: i32)  
    {
        
        todo!();
        /*
            if (data_size != input_size_) {
          // Since the best performance is obtained when the tensor is HW-packed, we
          // put X.size() to W.
          input_size_ = data_size;
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
              data_desc_,
              GetCudnnTensorFormat(StorageOrder::NCHW),
              data_type,
              1,
              1,
              1,
              input_size_));
        }
        */
    }
}

impl Drop for CudnnActivationOpBase {
    fn drop(&mut self) {
        todo!();
        //CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
        //CUDNN_ENFORCE(cudnnDestroyActivationDescriptor(act_desc_));
    }
}

pub struct CudnnActivationOp<CudnnActivationMode> {

    //USE_OPERATOR_FUNCTIONS(CUDAContext);

    base: CudnnActivationOpBase,
    phantom: PhantomData<CudnnActivationMode>,
}

impl<CudnnActivationMode> CudnnActivationOp<CudnnActivationMode> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : CudnnActivationOpBase(std::forward<Args>(args)...) 

        CUDNN_ENFORCE(cudnnSetActivationDescriptor(
            act_desc_, kCudnnActivationMode, CUDNN_PROPAGATE_NAN, 0.0));
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>() -> bool {
        todo!();
        /*
            const auto& X = Input(0);

            auto* Y = Output(0, X.sizes(), at::dtype<T>());
            if (X.numel() == 0) {
              Y->template mutable_data<T>();
              return true;
            }
            this->SetTensorDescriptor(cudnnTypeWrapper<T>::type, X.numel());
            CUDNN_ENFORCE(cudnnActivationForward(
                this->cudnn_wrapper_.inline_cudnn_handle(),
                this->act_desc_,
                cudnnTypeWrapper<T>::kOne(),
                this->data_desc_,
                X.template data<T>(),
                cudnnTypeWrapper<T>::kZero(),
                this->data_desc_,
                Y->template mutable_data<T>()));
            return true;
        */
    }
}

pub struct CudnnActivationGradientOp<CudnnActivationMode> {
    //USE_OPERATOR_FUNCTIONS(CUDAContext);
    base:    CudnnActivationOpBase,
    phantom: PhantomData<CudnnActivationMode>,
}

impl<CudnnActivationMode> CudnnActivationGradientOp<CudnnActivationMode> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : CudnnActivationOpBase(std::forward<Args>(args)...) 

        CUDNN_ENFORCE(cudnnSetActivationDescriptor(
            act_desc_, kCudnnActivationMode, CUDNN_PROPAGATE_NAN, 0.0));
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>() -> bool {
        todo!();
        /*
            const auto& Y = Input(0);
            const auto& dY = Input(1);

            auto* dX = Output(0, Y.sizes(), at::dtype<T>());
            if (Y.numel() == 0) {
              dX->template mutable_data<T>();
              return true;
            }
            this->SetTensorDescriptor(cudnnTypeWrapper<T>::type, Y.numel());
            CUDNN_ENFORCE(cudnnActivationBackward(
                this->cudnn_wrapper_.inline_cudnn_handle(),
                this->act_desc_,
                cudnnTypeWrapper<T>::kOne(),
                this->data_desc_,
                Y.template data<T>(),
                this->data_desc_,
                dY.template data<T>(),
                this->data_desc_,
                Y.template data<T>(), // Use Y_data as placeholder here.
                cudnnTypeWrapper<T>::kZero(),
                this->data_desc_,
                dX->template mutable_data<T>()));
            return true;
        */
    }
}
