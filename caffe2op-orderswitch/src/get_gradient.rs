crate::ix!();

pub struct GetNHWC2NCHWGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetNHWC2NCHWGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "NCHW2NHWC",
            "",
            std::vector<std::string>{GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

pub struct GetNCHW2NHWCGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetNCHW2NHWCGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "NHWC2NCHW",
            "",
            std::vector<std::string>{GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{NHWC2NCHW, GetNHWC2NCHWGradient}

register_gradient!{NCHW2NHWC, GetNCHW2NHWCGradient}
