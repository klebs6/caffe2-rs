crate::ix!();

pub struct GetGatherByKeyGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetGatherByKeyGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            ArgumentHelper argsHelper(def_);
        auto pack_first_input =
            argsHelper.GetSingleArgument<int>("pack_first_input", 0);

        Argument packArg = MakeArgument<int>("pack_first_input", pack_first_input);
        if (g_output_[0].IsDense()) {
          std::vector<std::string> inputs;
          for (int i = 1; i < g_input_.size(); ++i) {
            inputs.push_back("_" + GI(i) + "_keys");
            inputs.push_back(GI(i));
          }
          return SingleGradientDef(
              "Partition",
              "",
              std::vector<std::string>{I(0), GO(0)},
              inputs,
              std::vector<Argument>{packArg});
        } else {
          std::vector<std::string> inputs;
          for (int i = 1; i < g_input_.size(); ++i) {
            inputs.push_back("_" + GI_I(i) + "_keys");
            inputs.push_back(GI_I(i));
            inputs.push_back(GI_V(i));
          }
          return SingleGradientDef(
              "Partition",
              "",
              std::vector<std::string>{I(0), GO_I(0), GO_V(0)},
              inputs,
              std::vector<Argument>{packArg});
        }
        */
    }
}
