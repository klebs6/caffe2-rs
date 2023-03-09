crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SpaceBatchOpBase<Context> {
    storage:     OperatorStorage,
    context:     Context,
    pad:         i32,
    pad_t:       i32,
    pad_l:       i32,
    pad_b:       i32,
    pad_r:       i32,
    block_size:  i32,
    order:       StorageOrder,
}

impl<Context> SpaceBatchOpBase<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            pad_(this->template GetSingleArgument<int>("pad", 0)),
            pad_t_(this->template GetSingleArgument<int>("pad_t", pad_)),
            pad_l_(this->template GetSingleArgument<int>("pad", pad_)),
            pad_b_(this->template GetSingleArgument<int>("pad", pad_)),
            pad_r_(this->template GetSingleArgument<int>("pad", pad_)),
            block_size_(this->template GetSingleArgument<int>("block_size", 2)),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))) 

        CAFFE_ENFORCE(order_ == StorageOrder::NCHW);
        */
    }
}
