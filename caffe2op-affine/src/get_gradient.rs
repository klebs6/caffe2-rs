crate::ix!();

pub struct GetAffineChannelGradient;

impl GetGradientDefs for GetAffineChannelGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            ArgumentHelper arg_helper(def_);
        const bool is_learnable =
            arg_helper.GetSingleArgument("is_learnable", false);
        if (is_learnable) {
          return SingleGradientDef(
              "AffineChannelGradient",
              "",
              std::vector<std::string>{GO(0), I(0), I(1)},
              std::vector<std::string>{GI(0), GI(1), GI(2)});
        } else {
          return SingleGradientDef(
              "AffineChannelGradient",
              "",
              std::vector<std::string>{GO(0), I(1)},
              std::vector<std::string>{GI(0)});
        }
        */
    }
}

register_gradient!{
    AffineChannel, 
    GetAffineChannelGradient
}
