crate::ix!();

register_cpu_operator!{Col2Im, Col2ImOp<f32, CPUContext>}

pub struct GetCol2ImGradient;

impl GetGradientDefs for GetCol2ImGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Im2Col", "", std::vector<string>{GO(0)}, std::vector<string>{GI(0)});
        */
    }
}

register_gradient!{Col2Im, GetCol2ImGradient}

register_cuda_operator!{Col2Im, Col2ImOp<f32, CUDAContext>}
