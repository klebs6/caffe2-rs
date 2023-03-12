crate::ix!();

register_cpu_operator!{
    SubGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        SubFunctor<CPUContext>>}

num_inputs!{SubGradient, 3}

num_outputs!{SubGradient, 2}

tensor_inference_function!{SubGradient, ElementwiseGradientOpShapeInference}

allow_inplace!{SubGradient, vec![(0, 0), (0, 1)]}

pub struct GetSubGradient { }

impl GetGradientDefs for GetSubGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SubGradient",
            "",
            std::vector<std::string>{GO(0), I(0), I(1)},
            std::vector<std::string>{GI(0), GI(1)});
        */
    }
}

register_gradient!{Sub, GetSubGradient}

register_cuda_operator!{
    Sub,
    BinaryElementwiseOp<NumericTypes, CUDAContext, SubFunctor<CUDAContext>>
}

register_cuda_operator!{
    SubGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CUDAContext,
        SubFunctor<CUDAContext>>
}
