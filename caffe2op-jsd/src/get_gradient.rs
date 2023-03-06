crate::ix!();

pub struct GetBernoulliJSDGradient {}

impl GetGradientDefs for GetBernoulliJSDGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BernoulliJSDGradient",
            "",
            vector<string>{GO(0), I(0), I(1)},
            vector<string>{GI(0)});
        */
    }
}
