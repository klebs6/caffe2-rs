crate::ix!();

pub struct GetScaleGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetScaleGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // CopyArguments is true by default so the "scale" arg is going to be copied
        return SingleGradientDef(
            "Scale", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    Scale, 
    GetScaleGradient
}
