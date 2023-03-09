crate::ix!();

pub struct GetBatchToSpaceGradient<'a> {
    base: GradientMakerStorage<'a>,
}

register_gradient!{
    BatchToSpace, 
    GetBatchToSpaceGradient
}

impl<'a> GetGradientDefs for GetBatchToSpaceGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SpaceToBatch", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}
