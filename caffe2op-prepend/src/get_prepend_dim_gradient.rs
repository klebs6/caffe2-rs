crate::ix!();

pub struct GetPrependDimGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetPrependDimGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "MergeDim", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

impl<'a> CopyArguments for GetPrependDimGradient<'a> {

    /// Arguments are no longer needed in backprop.
    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}
