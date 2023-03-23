crate::ix!();

pub struct MIOPENActivationGradientOp<MIOPENActivationMode> {
    base:                        MIOPENActivationOpBase,
    phantomMIOPENActivationMode: PhantomData<MIOPENActivationMode>,
}

impl<MIOPENActivationMode> MIOPENActivationGradientOp<MIOPENActivationMode> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : MIOPENActivationOpBase(operator_def, ws) 
        MIOPEN_ENFORCE(miopenSetActivationDescriptor(
            act_desc_, kMIOPENActivationMode, 1.0, 1.0, 1.0));
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
            auto* dX = Output(0);
            dX->ResizeLike(Y);
            if (Y.size() == 0) {
              dX->template mutable_data<T>();
              return true;
            }
            // See if we need to reshape.
            if (Y.sizes() != mio_dims_) {
              VLOG(1) << "Setting descriptors.";
              mio_dims_ = Y.sizes().vec();
              int C = 1, H = 1, W = 1;
              if (Y.ndim() == 4) {
                // Normal 4-dimensional tensors for images.
                C = Y.dim32(1);
                H = Y.dim32(2);
                W = Y.dim32(3);
              } else {
                // If Y is not 4-dimensional, we will simply use H = 1 and W = 1
                // and wrap everything into C.
                C = Y.size() / Y.dim32(0);
              }
              MIOPEN_ENFORCE(miopenSet4dTensorDescriptor(
                  data_desc_, miopenTypeWrapper<T>::type, Y.dim32(0), C, H, W));
            }
            MIOPEN_ENFORCE(miopenActivationBackward(
                this->miopen_wrapper_.inline_miopen_handle(),
                this->act_desc_,
                miopenTypeWrapper<T>::kOne(),
                this->data_desc_,
                Y.template data<T>(),
                this->data_desc_,
                dY.template data<T>(),
                this->data_desc_,
                Y.template data<T>(),
                miopenTypeWrapper<T>::kZero(),
                this->data_desc_,
                dX->template mutable_data<T>()));
            return true;
        */
    }
}
