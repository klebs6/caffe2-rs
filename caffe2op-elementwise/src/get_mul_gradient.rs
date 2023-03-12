crate::ix!();

pub struct GetMulGradient;

impl GetGradientDefs for GetMulGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "MulGradient",
            "",
            std::vector<std::string>{GO(0), I(0), I(1)},
            std::vector<std::string>{GI(0), GI(1)});
        */
    }
}

register_gradient!{Mul, GetMulGradient}
