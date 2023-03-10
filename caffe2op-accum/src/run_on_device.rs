crate::ix!();

impl<T,Context> AccumulateOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            gamma_(static_cast<T>(
                this->template GetSingleArgument<float>("gamma", 1.0)))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);

        // TODO: the operator depends on output being set to 0 before the run
        auto* output = Output(0, input.sizes(), at::dtype<T>());
        math::Axpby<T, T, Context>(
            input.numel(),
            static_cast<T>(1),
            input.template data<T>(),
            gamma_,
            output->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}
