crate::ix!();

pub struct GetIntegralImageGradient {}

impl GetGradientDefs for GetIntegralImageGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "IntegralImageGradient",
            "",
            vector<string>{I(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}
