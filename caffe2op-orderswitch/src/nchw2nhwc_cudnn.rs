crate::ix!();

pub struct CudnnNCHW2NHWCOp {
    base: CudnnOrderSwithOpBase,
}

register_cudnn_operator!{
    NHWC2NCHW, 
    CudnnNHWC2NCHWOp
}

register_cuda_operator!{
    NHWC2NCHW, 
    NHWC2NCHWOp<f32, CUDAContext>
}

impl CudnnNCHW2NHWCOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : CudnnOrderSwithOpBase(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);

        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = X.dim32(1);
        const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
        std::vector<int> Y_dims(ndim);
        Y_dims[0] = N;
        Y_dims[ndim - 1] = C;
        std::copy(X_dims.cbegin() + 2, X_dims.cend(), Y_dims.begin() + 1);
        std::vector<int64_t> Y_dims_64;
        std::copy(Y_dims.cbegin(), Y_dims.cend(), std::back_inserter(Y_dims_64));
        auto* Y = Output(0, Y_dims_64, at::dtype<T>());
        if (cached_X_dims_ != X_dims) {
          cached_X_dims_ = X_dims;
          SetTensorDescriptor(
              cudnnTypeWrapper<T>::type, StorageOrder::NCHW, X_dims, X_desc_);
          SetTensorDescriptor(
              cudnnTypeWrapper<T>::type, StorageOrder::NHWC, Y_dims, Y_desc_);
        }
        CUDNN_ENFORCE(cudnnTransformTensor(
            cudnn_wrapper_.inline_cudnn_handle(),
            cudnnTypeWrapper<T>::kOne(),
            X_desc_,
            X.template data<T>(),
            cudnnTypeWrapper<T>::kZero(),
            Y_desc_,
            Y->template mutable_data<T>()));
        return true;
        */
    }
}
