crate::ix!();

pub struct GetTTSparseLengthsGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetTTSparseLengthsGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // set up the input and output
        return SingleGradientDef(
            "TTSparseLengthsSumGradient",
            "",
            // CORE0, CORE1, CORE2, LENGTHS, CORE0_output, CORE1_output,
            // indices, dY
            vector<string>{
                I(0), I(1), I(2), I(4), O(1), O(2), O(3), GO(0)},
            // dCore0, dCore1, dCore2
            vector<string>{GI(0), GI(1), GI(2)});
        */
    }
}

register_gradient!{
    TTSparseLengthsSum, 
    GetTTSparseLengthsGradient
}
