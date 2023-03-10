crate::ix!();

pub struct GetAbsGradient;

impl GetGradientDefs for GetAbsGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "AbsGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Abs, GetAbsGradient}
