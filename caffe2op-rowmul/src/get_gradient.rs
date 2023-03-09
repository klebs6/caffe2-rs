crate::ix!();

pub struct GetRowMulGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRowMulGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return vector<OperatorDef>{
            CreateOperatorDef(
                "RowMul", "", vector<string>{GO(0), I(1)}, vector<string>{GI(0)}),
            CreateOperatorDef(
                "Mul",
                "",
                vector<string>{GO(0), I(0)},
                vector<string>{GI(1) + "before_aggregate"}),
            CreateOperatorDef(
                "ReduceTailSum",
                "",
                vector<string>{GI(1) + "before_aggregate"},
                vector<string>{GI(1)})};
        */
    }
}

register_gradient!{
    RowMul, 
    GetRowMulGradient
}
