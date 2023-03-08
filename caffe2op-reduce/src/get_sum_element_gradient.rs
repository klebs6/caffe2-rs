crate::ix!();

pub struct GetSumElementsGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSumElementsGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SumElementsGradient",
            "",
            vector<string>{I(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{SumElements, GetSumElementsGradient}
