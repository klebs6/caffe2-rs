crate::ix!();

pub struct MaxGradientOp<T,Context> {
    base: SelectGradientOpBase<T, Context>,
    phantom: PhantomData<T>,
}

num_inputs!{MaxGradient, (3,INT_MAX)}

num_outputs!{MaxGradient, (1,INT_MAX)}

impl<T,Context> MaxGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : SelectGradientOpBase<T, Context>(std::forward<Args>(args)...)
        */
    }
}

register_cpu_operator!{MaxGradient, MaxGradientOp<float, CPUContext>}

pub struct GetMaxGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetMaxGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            std::vector<std::string> inputs = {O(0), GO(0)};
        std::vector<std::string> grad_inputs;
        for (int i = 0; i < def_.input_size(); ++i) {
          inputs.push_back(I(i));
          grad_inputs.push_back(GI(i));
        }
        return SingleGradientDef("MaxGradient", "", inputs, grad_inputs);
        */
    }
}

register_gradient!{Max, GetMaxGradient}
