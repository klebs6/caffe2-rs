crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AffineChannelGradientOp<T, Context> {
    storage:      OperatorStorage,
    context:      Context,
    order:        StorageOrder,
    is_learnable: bool,
    phantom:      PhantomData<T>,
}

num_inputs!{AffineChannelGradient,    (2,3)}

num_outputs!{AffineChannelGradient,   (1,3)}

allow_inplace!{AffineChannelGradient, vec![(0, 0)]}

register_cpu_operator!{
    AffineChannelGradient,
    AffineChannelGradientOp<f32, CPUContext>
}

impl<T,Context> AffineChannelGradientOp<T, Context> {
    
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
}
