crate::ix!();

pub struct GetTopKGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetTopKGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "TopKGradient",
            "",
            vector<string>{GO(0), O(1), I(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{TopK, GetTopKGradient}
