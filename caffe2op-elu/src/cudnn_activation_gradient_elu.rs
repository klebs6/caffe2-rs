crate::ix!();

#[USE_OPERATOR_FUNCTIONS("CUDAContext")]
pub struct CudnnActivationGradientOpELU {
    base:  CudnnActivationOpBase,
    alpha: f32,
}

register_cudnn_operator!{
    EluGradient,
    CudnnActivationGradientOp<CUDNN_ACTIVATION_ELU>
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
