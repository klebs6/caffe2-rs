crate::ix!();

pub struct GetTanGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetTanGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
                "TanGradient",
                "",
                std::vector<std::string>{I(0), GO(0)},
                std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Tan, GetTanGradient}
