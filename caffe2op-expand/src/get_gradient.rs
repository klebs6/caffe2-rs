crate::ix!();

pub struct GetSqueezeGradient;

impl GetGradientDefs for GetSqueezeGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ExpandDims", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

pub struct GetExpandDimsGradient;

impl GetGradientDefs for GetExpandDimsGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Squeeze", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}
