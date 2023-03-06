crate::ix!();

pub struct GetInstanceNormGradient { }

impl GetGradientDefs for GetInstanceNormGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            std::vector<std::string> inputs = {I(0), I(1), I(2), GO(0)};
        if (def_.output_size() >= 2) {
          inputs.push_back(O(1));
        }
        if (def_.output_size() >= 3) {
          inputs.push_back(O(2));
        }
        return SingleGradientDef(
            "InstanceNormGradient",
            "",
            inputs,
            std::vector<std::string>({GI(0), GI(1), GI(2)}));
        */
    }
}
