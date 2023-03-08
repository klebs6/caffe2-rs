crate::ix!();

impl<Context> ReplaceNaNOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            T value = this->template GetSingleArgument<T>("value", 0);

        auto& input = Input(0);

        auto* output = Output(0, input.sizes(), at::dtype<T>());

        const T* input_data = input.template data<T>();
        T* output_data = output->template mutable_data<T>();

        ReplaceNaN<T>(value, input.numel(), input_data, output_data);

        return true;
        */
    }
}
