crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LRNOpBase<T, Context> {
    storage: OperatorStorage,
    context: Context,
    size:    i32,
    alpha:   f32,
    beta:    f32,
    bias:    f32,
    order:   StorageOrder,
    pre_pad: i32,

    /**
      | Input: X; Output: Y, scale.
      |
      */
    phantom: PhantomData<T>,
}

impl<T, Context> LRNOpBase<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            size_(this->template GetSingleArgument<int>("size", 0)),
            alpha_(this->template GetSingleArgument<float>("alpha", 0)),
            beta_(this->template GetSingleArgument<float>("beta", 0)),
            bias_(this->template GetSingleArgument<float>("bias", 1)),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))),
            pre_pad_((size_ - 1) / 2) 

        DCHECK_GT(size_, 0);
        DCHECK_EQ(size_ % 2, 1);
        DCHECK_GT(alpha_, 0);
        DCHECK_GT(beta_, 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            switch (order_) {
          case StorageOrder::NHWC:
            return RunOnDeviceWithOrderNHWC();
          case StorageOrder::NCHW:
            return RunOnDeviceWithOrderNCHW();
          default:
            LOG(FATAL) << "Unknown storage order: " << order_;
        }
        // To suppress old compiler warnings
        return true;
        */
    }
}
