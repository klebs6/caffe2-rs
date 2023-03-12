crate::ix!();

pub struct GetCosGradient {}

impl GetGradientDefs for GetCosGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "CosGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Cos, GetCosGradient}
