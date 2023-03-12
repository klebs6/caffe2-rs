crate::ix!();

pub struct GetCopyGradient;

impl GetGradientDefs for GetCopyGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "CopyOnDeviceLike",
            "",
            vector<string>{GO(0), I(0)},
            vector<string>{GI(0)});
        */
    }
}
