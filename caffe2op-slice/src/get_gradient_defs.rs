crate::ix!();

pub struct GetSliceGradient {

}

impl GetGradientDefs for GetSliceGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (def_.input_size() > 1) {
          return vector<OperatorDef>{CreateOperatorDef(
              "SliceGradient",
              "",
              std::vector<string>{I(0), I(1), I(2), GO(0)},
              std::vector<string>{GI(0)})};
        } else {
          return vector<OperatorDef>{CreateOperatorDef(
              "SliceGradient",
              "",
              std::vector<string>{I(0), GO(0)},
              std::vector<string>{GI(0)})};
        }
        */
    }
}

register_gradient!{Slice, GetSliceGradient}
