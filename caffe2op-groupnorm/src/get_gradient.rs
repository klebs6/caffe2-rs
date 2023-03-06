crate::ix!();

/**
  | Warning: mu and rsig are for backward usage or
  | reference. They should NOT be used as forward
  | activations as they have no direct gradients
  | computed.
  */
pub struct GetGroupNormGradient;

impl GetGradientDefs for GetGroupNormGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "GroupNormGradient",
            "",
            std::vector<std::string>{GO(0), I(0), I(1), I(2), O(1), O(2)},
            std::vector<std::string>{GI(0), GI(1), GI(2)});
        */
    }
}

register_gradient!{GroupNorm, GetGroupNormGradient}
