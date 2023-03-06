crate::ix!();

pub struct GetLogitGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLogitGradient<'a> {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return vector<OperatorDef>{CreateOperatorDef(
            "LogitGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)})};
        */
    }
}

register_gradient!{Logit, GetLogitGradient}
