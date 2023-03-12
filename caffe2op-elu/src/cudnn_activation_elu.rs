crate::ix!();

///------------------------------
#[USE_OPERATOR_FUNCTIONS("CUDAContext")]
pub struct CudnnActivationOpELU {
    base:  CudnnActivationOpBase,
    alpha: f32,
}

register_cudnn_operator!{
    Elu, 
    CudnnActivationOp<CUDNN_ACTIVATION_ELU>
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
