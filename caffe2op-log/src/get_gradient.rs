crate::ix!();

pub struct GetLogGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLogGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Div",
            "",
            std::vector<std::string>{GO(0), I(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Log, GetLogGradient}

register_cuda_operator!{
    Log,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CUDAContext,
        LogFunctor<CUDAContext>>
}
