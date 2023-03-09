crate::ix!();

pub struct GetSqrtGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSqrtGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            Argument scale_arg;
        scale_arg.set_name("scale");
        scale_arg.set_f(0.5);
        return std::vector<OperatorDef>{CreateOperatorDef(
                                            "Scale",
                                            "",
                                            std::vector<std::string>{GO(0)},
                                            std::vector<std::string>{GI(0)},
                                            std::vector<Argument>{scale_arg}),
                                        CreateOperatorDef(
                                            "Div",
                                            "",
                                            std::vector<std::string>{GI(0), O(0)},
                                            std::vector<std::string>{GI(0)})};
        */
    }
}

register_gradient!{Sqrt, GetSqrtGradient}

register_cuda_operator!{
    Sqrt,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        SqrtFunctor<CUDAContext>>
}
