crate::ix!();

pub struct GetLengthsTopKGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLengthsTopKGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "LengthsTopKGradient",
            "",
            vector<string>{I(1), O(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{LengthsTopK, GetLengthsTopKGradient}
