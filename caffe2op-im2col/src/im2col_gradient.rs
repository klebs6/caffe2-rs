crate::ix!();

pub struct GetIm2ColGradient;

impl GetGradientDefs for GetIm2ColGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Col2Im",
            "",
            std::vector<string>{GO(0), I(0)},
            std::vector<string>{GI(0)});
        */
    }
}

register_gradient!{Im2Col, GetIm2ColGradient}
