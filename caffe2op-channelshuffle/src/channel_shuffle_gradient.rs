crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ChannelShuffleGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    order:   StorageOrder,
    group:   i32,
    phantom: PhantomData<T>,
}

num_inputs!{ChannelShuffleGradient, 1}

num_outputs!{ChannelShuffleGradient, 1}

identical_type_and_shape!{ChannelShuffleGradient}

impl<T,Context> ChannelShuffleGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<std::string>("order", "NCHW"))),
            OP_SINGLE_ARG(int, "group", group_, 1) 

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
