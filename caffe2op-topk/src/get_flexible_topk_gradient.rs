crate::ix!();

pub struct GetFlexibleTopKGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a>GetGradientDefs for GetFlexibleTopKGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "FlexibleTopKGradient",
            "",
            vector<string>{I(0), I(1), GO(0), O(1)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    FlexibleTopK, 
    GetFlexibleTopKGradient
}
