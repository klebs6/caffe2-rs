crate::ix!();

register_cpu_operator!{
    ConvTransposeGradient,
    ConvTransposeGradientOp<f32, CPUContext>
}

num_inputs!{ConvTransposeGradient, 3}

num_outputs!{ConvTransposeGradient, (1,3)}

pub struct GetConvTransposeGradient;

impl GetGradientDefs for GetConvTransposeGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            auto compute_dX =
            !ArgumentHelper::GetSingleArgument(def_, "no_gradient_to_input", false);

        CAFFE_ENFORCE(3 == def_.input_size() || 2 == def_.input_size());
        if (def_.input_size() == 3 && compute_dX) {
          return SingleGradientDef(
              "ConvTransposeGradient",
              "",
              vector<string>{I(0), I(1), GO(0)},
              vector<string>{GI(1), GI(2), GI(0)});
        } else if (def_.input_size() == 3) {
          return SingleGradientDef(
              "ConvTransposeGradient",
              "",
              vector<string>{I(0), I(1), GO(0)},
              vector<string>{GI(1), GI(2)});
        } else if (compute_dX) {
          return SingleGradientDef(
              "ConvTransposeGradient",
              "",
              vector<string>{I(0), I(1), GO(0)},
              vector<string>{GI(1), GI(0)},
              vector<Argument>{MakeArgument<bool>("no_bias", true)});
        } else {
          return SingleGradientDef(
              "ConvTransposeGradient",
              "",
              vector<string>{I(0), I(1), GO(0)},
              vector<string>{GI(1)},
              vector<Argument>{MakeArgument<bool>("no_bias", true)});
        }
        */
    }
}

register_gradient!{
    ConvTranspose, 
    GetConvTransposeGradient
}
