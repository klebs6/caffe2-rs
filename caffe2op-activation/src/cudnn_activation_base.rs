crate::ix!();

#[USE_OPERATOR_FUNCTIONS("CUDAContext")]
pub struct CudnnActivationOpBase {
    storage:       OperatorStorage,
    context:       CUDAContext,
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
