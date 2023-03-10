crate::ix!();

pub struct GetAtanGradient;

impl GetGradientDefs for GetAtanGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "AtanGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Atan, GetAtanGradient}
