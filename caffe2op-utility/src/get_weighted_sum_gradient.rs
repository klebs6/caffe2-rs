crate::ix!();

pub struct GetWeightedSumGradient;

impl GetGradientDefs for GetWeightedSumGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            ArgumentHelper argsHelper(def_);
        const bool grad_on_w = argsHelper.GetSingleArgument<bool>("grad_on_w", 0);

        auto inputs = vector<string>{GO(0)};
        auto outputs = vector<string>();
        for (int i = 0; i < def_.input_size(); i += 2) {
          inputs.push_back(I(i));
          inputs.push_back(I(i + 1));
          outputs.push_back(GI(i));
        }

        if (grad_on_w) {
          for (int i = 0; i < def_.input_size(); i += 2) {
            outputs.push_back(GI(i + 1));
          }
        }

        return SingleGradientDef("WeightedSumGradient", "", inputs, outputs);
        */
    }
}
