crate::ix!();

pub struct GetRecurrentNetworkGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRecurrentNetworkGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            ArgumentHelper argsHelper(def_);
        auto params = argsHelper.GetRepeatedArgument<int32_t>("param");
        auto recurrentInputs =
            argsHelper.GetRepeatedArgument<int32_t>("initial_recurrent_state_ids");

        std::vector<std::string> gradientInputs;

        // Argument specifies which outputs have external gradient, (0) by default
        auto outputs_with_grads =
            argsHelper.GetRepeatedArgument<int32_t>("outputs_with_grads");
        CAFFE_ENFORCE(outputs_with_grads.size() > 0);
        for (auto id : outputs_with_grads) {
          gradientInputs.push_back(GO(id));
        }

        // All inputs and outputs are passed back
        for (int i = 0; i < def_.input_size(); ++i) {
          gradientInputs.push_back(I(i));
        }
        for (int i = 0; i < def_.output_size(); ++i) {
          gradientInputs.push_back(O(i));
        }

        // We calculate gradients only for parameters and recurrent inputs
        std::vector<std::string> gradientOutputs;
        gradientOutputs.push_back(GI(0));
        for (auto id : params) {
          gradientOutputs.push_back(GI(id));
        }
        for (auto id : recurrentInputs) {
          gradientOutputs.push_back(GI(id));
        }

        VLOG(1) << "Gradient blobs: " << c10::Join(", ", gradientOutputs);

        return SingleGradientDef(
            "RecurrentNetworkGradient", "", gradientInputs, gradientOutputs);
        */
    }
}

register_gradient!{RecurrentNetwork, GetRecurrentNetworkGradient}
