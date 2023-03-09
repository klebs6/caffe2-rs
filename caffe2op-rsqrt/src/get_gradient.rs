crate::ix!();

pub struct GetRsqrtGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRsqrtGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "RsqrtGradient",
            "",
            std::vector<std::string>{GO(0), O(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{
    Rsqrt, 
    GetRsqrtGradient
}
