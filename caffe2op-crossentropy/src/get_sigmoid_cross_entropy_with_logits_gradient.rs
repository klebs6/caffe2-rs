crate::ix!();

pub struct GetSigmoidCrossEntropyWithLogitsGradient;

impl GetGradientDefs for GetSigmoidCrossEntropyWithLogitsGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SigmoidCrossEntropyWithLogitsGradient",
            "",
            vector<string>{GO(0), I(0), I(1)},
            vector<string>{GI(0)});
        */
    }
}
