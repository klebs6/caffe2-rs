crate::ix!();

pub struct GetFloatToHalfGradient ;

impl GetGradientDefs for GetFloatToHalfGradient {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "HalfToFloat", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{FloatToHalf, GetFloatToHalfGradient}

no_gradient!{Float16ConstantFill}

