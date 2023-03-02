crate::ix!();

pub struct GetExpGradient;

impl GetGradientDefs for GetExpGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Mul",
            "",
            std::vector<std::string>{O(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Exp, GetExpGradient}

register_cuda_operator!{
    Exp,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CUDAContext,
        ExpFunctor<CUDAContext>>
}

