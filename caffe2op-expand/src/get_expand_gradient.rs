crate::ix!();

pub struct GetExpandGradient;

impl GetGradientDefs for GetExpandGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ExpandGradient",
            "",
            std::vector<string>{GO(0), I(0)},
            std::vector<string>{GI(0)});
        */
    }
}

register_gradient!{Expand, GetExpandGradient}
