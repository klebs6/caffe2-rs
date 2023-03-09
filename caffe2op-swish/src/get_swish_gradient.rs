crate::ix!();

pub struct GetSwishGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSwishGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
                        "SwishGradient",
                        "",
                        std::vector<std::string>{I(0), O(0), GO(0)},
                        std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Swish, GetSwishGradient}
