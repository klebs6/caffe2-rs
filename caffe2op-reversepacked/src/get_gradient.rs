crate::ix!();

pub struct GetReversePackedSegsGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReversePackedSegsGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ReversePackedSegs",
            "",
            vector<string>{GO(0), I(1)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{ReversePackedSegs, GetReversePackedSegsGradient}
