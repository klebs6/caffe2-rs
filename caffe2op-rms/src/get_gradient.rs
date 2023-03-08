crate::ix!();

pub struct GetRMSNormGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRMSNormGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "RMSNormGradient",
            "",
            std::vector<std::string>{GO(0), I(0), I(1), O(1)},
            std::vector<std::string>{GI(0), GI(1), GI(2)});
        */
    }
}

register_gradient!{RMSNorm, GetRMSNormGradient}
