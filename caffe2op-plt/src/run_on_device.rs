crate::ix!();

pub type PiecewiseLinearTransformOpFloatCPU = PiecewiseLinearTransformOp<f32, CPUContext>;

impl<T, Context> PiecewiseLinearTransformOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        binary_ = this->template GetSingleArgument<bool>("binary", false);

        // Retrieve transform params (i.e., the linear functions).
        bounds_from_arg_ = this->template GetRepeatedArgument<T>("bounds");
        slopes_from_arg_ = this->template GetRepeatedArgument<T>("slopes");
        intercepts_from_arg_ = this->template GetRepeatedArgument<T>("intercepts");
        transform_param_from_arg_ = CheckTransParamFromArg();
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return binary_ ? TransformBinary() : TransformGeneral();
        */
    }
}

register_cpu_operator!{
    PiecewiseLinearTransform,
    PiecewiseLinearTransformOp<f32, CPUContext>
}

should_not_do_gradient!{PiecewiseLinearTransform}
