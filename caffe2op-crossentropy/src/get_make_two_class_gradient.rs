crate::ix!();

pub struct GetMakeTwoClassGradient;

impl GetGradientDefs for GetMakeTwoClassGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "MakeTwoClassGradient",
            "",
            vector<string>{GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_cpu_operator!{
    MakeTwoClassGradient, 
    MakeTwoClassGradientOp<f32, CPUContext>
}
