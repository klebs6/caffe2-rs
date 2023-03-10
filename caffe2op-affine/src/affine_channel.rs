crate::ix!();

/**
  | Applies a separate affine transformation
  | to each channel of the input. Useful
  | for replacing spatial batch norm with
  | its equivalent fixed transformation.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AffineChannelOp<T, Context> {
    storage:      OperatorStorage,
    context:      Context,
    order:        StorageOrder,
    is_learnable: bool,
    phantom:      PhantomData<T>,
}

num_inputs!{AffineChannel, 3}

num_outputs!{AffineChannel, 1}

inputs!{AffineChannel, 
    0 => ("X", "Feature map input with order NCHW or NHWC."),
    1 => ("scale", "1D input of shape (C); the c-th element is the scale factor of the affine transformation for the c-th channel of the input."),
    2 => ("bias", "1D input of shape (C); the c-th element is the bias of the affine transformation for the c-th channel of the input.")
}

outputs!{AffineChannel, 
    0 => ("Y", "Output with the same order of Input.")
}

allow_inplace!{AffineChannel, vec![(0, 0)]}

register_cpu_operator!{AffineChannel,          AffineChannelOp<f32, CPUContext>}

impl<T,Context> AffineChannelOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
          order_(StringToStorageOrder(
                  this->template GetSingleArgument<std::string>("order", "NCHW"))),
                  OP_SINGLE_ARG(bool, "is_learnable", is_learnable_, false) 

                      CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return order_ == StorageOrder::NCHW ? RunOnDeviceWithOrderNCHW()
                                            : RunOnDeviceWithOrderNHWC();
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
        const auto& X = Input(0);
        const auto& scale = Input(1);
        const auto& bias = Input(2);

        if (is_learnable_) {
          CAFFE_ENFORCE(
              !IsInputOutputAlias(0, 0),
              "In-place affine_channel_op is not supported when "
              "is_learnable = true.");
        }
        const int N = X.dim32(0);
        const int C = X.dim32(1);
        const int HxW = X.numel() / (N * C);
        auto* Y = Output(0, X.sizes(), at::dtype<T>());
        math::AffineChannel<T, Context, StorageOrder::NCHW>(
            N,
            C,
            HxW,
            X.template data<T>(),
            scale.template data<T>(),
            bias.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        const auto& scale = Input(1);
        const auto& bias = Input(2);

        if (is_learnable_) {
          CAFFE_ENFORCE(
              !IsInputOutputAlias(0, 0),
              "In-place affine_channel_op is not supported when "
              "is_learnable = true.");
        }
        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = X.dim32(ndim - 1);
        const int HxW = X.numel() / (N * C);
        auto* Y = Output(0, X.sizes(), at::dtype<T>());
        math::AffineChannel<T, Context, StorageOrder::NHWC>(
            N,
            C,
            HxW,
            X.template data<T>(),
            scale.template data<T>(),
            bias.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}
