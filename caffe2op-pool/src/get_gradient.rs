crate::ix!();

pub struct GetPoolGradient;

impl GetGradientDefs for GetPoolGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            std::vector<std::string>{I(0), O(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}
