crate::ix!();

pub struct MinGradientOp<T,Context> {
    base: SelectGradientOpBase<T, Context>,
    phantom: PhantomData<T>,
}

num_inputs!{MinGradient, (3,INT_MAX)}

num_outputs!{MinGradient, (1,INT_MAX)}

impl<T,Context> MinGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : SelectGradientOpBase<T, Context>(std::forward<Args>(args)...)
        */
    }
}

register_cpu_operator!{MinGradient, MinGradientOp<float, CPUContext>}

pub struct GetMinGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetMinGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            std::vector<std::string> inputs = {O(0), GO(0)};
        std::vector<std::string> grad_inputs;
        for (int i = 0; i < def_.input_size(); ++i) {
          inputs.push_back(I(i));
          grad_inputs.push_back(GI(i));
        }
        return SingleGradientDef("MinGradient", "", inputs, grad_inputs);
        */
    }
}

register_gradient!{Min, GetMinGradient}
