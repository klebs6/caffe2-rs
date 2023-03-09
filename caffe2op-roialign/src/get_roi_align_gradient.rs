crate::ix!();

pub struct GetRoIAlignGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRoIAlignGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "RoIAlignGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{RoIAlign, GetRoIAlignGradient}

pub type RoIAlignGradientCPUOp<T> = RoIAlignGradientOp<T, CPUContext>;
