crate::ix!();

pub struct GetSpaceToBatchGradient<'a> {
    base: GradientMakerStorage<'a>,
}

register_gradient!{
    SpaceToBatch, 
    GetSpaceToBatchGradient
}

impl<'a> GetGradientDefs for GetSpaceToBatchGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BatchToSpace", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}
