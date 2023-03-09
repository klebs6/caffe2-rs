crate::ix!();

pub struct GetRoIPoolGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRoIPoolGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "RoIPoolGradient",
            "",
            vector<string>{I(0), I(1), O(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    RoIPool, 
    GetRoIPoolGradient
}
