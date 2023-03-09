crate::ix!();

pub struct GetSquareRootDivideGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSquareRootDivideGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SquareRootDivide",
            "",
            vector<string>{GO(0), I(1)},
            vector<string>{GI(0)});
        */
    }
}
