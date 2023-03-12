crate::ix!();

pub struct GetLabelCrossEntropyGradient;

impl GetGradientDefs for GetLabelCrossEntropyGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "LabelCrossEntropyGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    LabelCrossEntropy, 
    GetLabelCrossEntropyGradient
}
