crate::ix!();

impl<Context> FlattenOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axis_(this->template GetSingleArgument<int>("axis", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        CAFFE_ENFORCE_GE(
            input.dim(), axis_, "The rank of the tensor must be >= axis.");
        output->Resize(input.size_to_dim(axis_), input.size_from_dim(axis_));
        context_.CopyItemsSameDevice(
            input.dtype(),
            input.numel(),
            input.raw_data(),
            output->raw_mutable_data(input.dtype()));
        return true;
        */
    }
}
