crate::ix!();

pub struct GetMomentsGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetMomentsGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "MomentsGradient",
            "",
            std::vector<std::string>{GO(0), GO(1), I(0), O(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Moments, GetMomentsGradient}
