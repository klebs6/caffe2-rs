crate::ix!();

pub struct GetLocallyConnectedGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLocallyConnectedGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE(def_.input_size() == 3 || def_.input_size() == 2);
        ArgumentHelper argsHelper(def_);
        const bool compute_dX =
            !argsHelper.GetSingleArgument<bool>("no_gradient_to_input", 0);

        if (def_.input_size() == 3) {
          if (compute_dX) {
            return SingleGradientDef(
                def_.type() + "Gradient",
                "",
                std::vector<string>{I(0), I(1), GO(0)},
                std::vector<string>{GI(1), GI(2), GI(0)});
          } else {
            return SingleGradientDef(
                def_.type() + "Gradient",
                "",
                std::vector<string>{I(0), I(1), GO(0)},
                std::vector<string>{GI(1), GI(2)});
          }
        } else {
          if (compute_dX) {
            return SingleGradientDef(
                def_.type() + "Gradient",
                "",
                std::vector<string>{I(0), I(1), GO(0)},
                std::vector<string>{GI(1), GI(0)},
                std::vector<Argument>{MakeArgument<int>("no_bias", 1)});
          } else {
            return SingleGradientDef(
                def_.type() + "Gradient",
                "",
                std::vector<string>{I(0), I(1), GO(0)},
                std::vector<string>{GI(1)},
                std::vector<Argument>{MakeArgument<int>("no_bias", 1)});
          }
        }
        */
    }
}
