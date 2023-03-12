crate::ix!();

pub struct GetWeightedSigmoidCrossEntropyWithLogitsGradient;

impl GetGradientDefs for GetWeightedSigmoidCrossEntropyWithLogitsGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "WeightedSigmoidCrossEntropyWithLogitsGradient",
            "",
            vector<string>{GO(0), I(0), I(1), I(2)},
            vector<string>{GI(0)});
        */
    }
}
