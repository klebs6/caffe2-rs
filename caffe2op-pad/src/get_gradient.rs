crate::ix!();

pub struct GetPadImageGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetPadImageGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "PadImageGradient", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{PadImage, GetPadImageGradient}
