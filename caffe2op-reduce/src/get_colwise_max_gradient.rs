crate::ix!();

num_inputs!{ColumnMaxGradient, 3}

num_outputs!{ColumnMaxGradient, 1}

pub struct GetColwiseMaxGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetColwiseMaxGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ColwiseMaxGradient",
            "",
            vector<string>{I(0), O(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{ColwiseMax, GetColwiseMaxGradient}
