crate::ix!();

pub struct GetElementwiseLinearGradient;

impl GetGradientDefs for GetElementwiseLinearGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
          "ElementwiseLinearGradient",
          "",
          vector<string>{GO(0), I(0), I(1)},
          vector<string>{GI(0), GI(1), GI(2)});
        */
    }
}

register_gradient!{
    ElementwiseLinear,
    GetElementwiseLinearGradient
}
