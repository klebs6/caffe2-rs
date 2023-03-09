crate::ix!();

pub struct GetTanhGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetTanhGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "TanhGradient",
            "",
            std::vector<std::string>{O(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{
    Tanh, 
    GetTanhGradient
}

register_cudnn_operator!{
    Tanh,
    CudnnActivationOp<CUDNN_ACTIVATION_TANH>
}

register_cudnn_operator!{
    TanhGradient, 
    CudnnActivationGradientOp<CUDNN_ACTIVATION_TANH>
}
