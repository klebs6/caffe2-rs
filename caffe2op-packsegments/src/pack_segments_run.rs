crate::ix!();

impl<Context> PackSegmentsOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            max_length_(this->template GetSingleArgument<int>("max_length", -1)),
            pad_minf_(this->template GetSingleArgument<bool>("pad_minf", false)),
            return_presence_mask_(this->template GetSingleArgument<bool>( "return_presence_mask", false)) 

        if (pad_minf_) {
          padding_ = -1.0 * std::numeric_limits<float>::infinity();
        } else {
          padding_ = 0;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int, long>>::call(this, Input(LENGTHS));
        */
    }
}
