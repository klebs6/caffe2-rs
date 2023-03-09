crate::ix!();

pub struct GetSigmoidGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSigmoidGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SigmoidGradient",
            "",
            std::vector<std::string>{O(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Sigmoid, GetSigmoidGradient}

register_cudnn_operator!{
    Sigmoid,         
    CudnnActivationOp<CUDNN_ACTIVATION_SIGMOID>
}

register_cudnn_operator!{
    SigmoidGradient, 
    CudnnActivationGradientOp<CUDNN_ACTIVATION_SIGMOID>
}
