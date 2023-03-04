crate::ix!();

pub struct GetHalfToFloatGradient ;

impl GetGradientDefs for GetHalfToFloatGradient {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "FloatToHalf", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{HalfToFloat, GetHalfToFloatGradient}
