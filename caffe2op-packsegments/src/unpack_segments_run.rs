crate::ix!();

impl<Context> UnpackSegmentsOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            max_length_(this->template GetSingleArgument<int>("max_length", -1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int, long>>::call(this, Input(LENGTHS));
        */
    }
}
