crate::ix!();

impl<Context> ModOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        divisor_ = this->template GetSingleArgument<int64_t>("divisor", 0);
        CAFFE_ENFORCE_NE(divisor_, 0, "divisor must not be 0");
        sign_follow_divisor_ =
            this->template GetSingleArgument<bool>("sign_follow_divisor", false);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(DATA));
        */
    }
}
