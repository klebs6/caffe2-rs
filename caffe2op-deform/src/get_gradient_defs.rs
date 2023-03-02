crate::ix!();

pub struct GetDeformConvGradient;

impl GetGradientDefs for GetDeformConvGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE(def_.input_size() == 3 || def_.input_size() == 4);

        ArgumentHelper argsHelper(def_);

        auto compute_dX =
            !argsHelper.GetSingleArgument<bool>("no_gradient_to_input", 0);

        if (def_.input_size() == 4) {
          if (compute_dX) {
            return SingleGradientDef(
                "DeformConvGradient",
                "",
                vector<string>{I(0), I(1), I(2), GO(0)},
                vector<string>{GI(1), GI(2), GI(3), GI(0)});
          } else {
            return SingleGradientDef(
                "DeformConvGradient",
                "",
                vector<string>{I(0), I(1), I(2), GO(0)},
                vector<string>{GI(1), GI(2), GI(3)});
          }
        } else {
          if (compute_dX) {
            return SingleGradientDef(
                "DeformConvGradient",
                "",
                vector<string>{I(0), I(1), I(2), GO(0)},
                vector<string>{GI(1), GI(2), GI(0)},
                vector<Argument>{MakeArgument<int>("no_bias", 1)});
          } else {
            return SingleGradientDef(
                "DeformConvGradient",
                "",
                vector<string>{I(0), I(1), I(2), GO(0)},
                vector<string>{GI(1), GI(2)},
                vector<Argument>{MakeArgument<int>("no_bias", 1)});
          }
        }
        */
    }
}

register_gradient!{
    DeformConv, 
    GetDeformConvGradient
}
