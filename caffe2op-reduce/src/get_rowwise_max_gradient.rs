crate::ix!();

num_inputs!{RowwiseMaxGradient, 3}

num_outputs!{RowwiseMaxGradient, 1}

pub struct GetRowwiseMaxGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRowwiseMaxGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "RowwiseMaxGradient",
            "",
            vector<string>{I(0), O(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{RowwiseMax, GetRowwiseMaxGradient}

