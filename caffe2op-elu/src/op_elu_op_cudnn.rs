crate::ix!();

///------------------------------
pub struct CudnnActivationOpELU {
    //USE_OPERATOR_FUNCTIONS(CUDAContext);
    base: CudnnActivationOpBase,

    alpha: f32,
}

impl CudnnActivationOpELU {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : CudnnActivationOpBase(std::forward<Args>(args)...),
            OP_SINGLE_ARG(float, "alpha", alpha_, 1.0f) 

        CUDNN_ENFORCE(cudnnSetActivationDescriptor(
            act_desc_,
            CUDNN_ACTIVATION_ELU,
            CUDNN_PROPAGATE_NAN,
            static_cast<double>(alpha_)));
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

///------------------------------
pub struct CudnnActivationGradientOpELU {

    //USE_OPERATOR_FUNCTIONS(CUDAContext);
    base: CudnnActivationOpBase,

    alpha: f32,
}

impl CudnnActivationGradientOpELU {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : CudnnActivationOpBase(std::forward<Args>(args)...),
            OP_SINGLE_ARG(float, "alpha", alpha_, 1.0f) 

        CUDNN_ENFORCE(cudnnSetActivationDescriptor(
            act_desc_,
            CUDNN_ACTIVATION_ELU,
            CUDNN_PROPAGATE_NAN,
            static_cast<double>(alpha_)));
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

register_cudnn_operator!{
    Elu, 
    CudnnActivationOp<CUDNN_ACTIVATION_ELU>
}

register_cudnn_operator!{
    EluGradient,
    CudnnActivationGradientOp<CUDNN_ACTIVATION_ELU>
}
