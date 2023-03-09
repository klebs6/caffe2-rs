crate::ix!();

pub struct GetSparseToDenseGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSparseToDenseGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Gather", "", vector<string>{GO(0), I(0)}, vector<string>{GI(1)});
        */
    }
}

register_gradient!{
    SparseToDense, 
    GetSparseToDenseGradient
}
