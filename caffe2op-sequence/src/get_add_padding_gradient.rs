crate::ix!();

pub struct GetAddPaddingGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetAddPaddingGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // whether to provide lengths as input to gradient
        vector<std::string> g_inputs{GO(0)};
        if (Def().input_size() > 1) {
          CAFFE_ENFORCE(Def().output_size() > 1);
          g_inputs.push_back(O(1));
        }

        vector<OperatorDef> ops;
        // gradient on the data
        ops.push_back(CreateOperatorDef(
            "RemovePadding", "", g_inputs, vector<string>{GI(0)}));
        // gradient on the start_padding (and end_padding)
        if (Def().input_size() >= 3) {
          std::vector<string> padding_grads{GI(2)};
          if (Def().input_size() == 4) {
            padding_grads.push_back(GI(3));
          }
          auto g_inputs2 = g_inputs;
          ops.push_back(
              CreateOperatorDef("GatherPadding", "", g_inputs2, padding_grads));
        }
        return ops;
        */
    }
}

register_gradient!{AddPadding, GetAddPaddingGradient}
