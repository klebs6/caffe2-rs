crate::ix!();

pub struct GetCrossEntropyGradient;

impl GetGradientDefs for GetCrossEntropyGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "CrossEntropyGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{CrossEntropy, GetCrossEntropyGradient}
