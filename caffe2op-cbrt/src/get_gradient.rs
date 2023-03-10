crate::ix!();

pub struct GetCbrtGradient;

impl GetGradientDefs for GetCbrtGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "CbrtGradient",
            "",
            std::vector<std::string>{GO(0), O(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Cbrt, GetCbrtGradient}
