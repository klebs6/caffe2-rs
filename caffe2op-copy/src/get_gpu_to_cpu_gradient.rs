crate::ix!();

pub struct GetGPUToCPUGradient;

impl GetGradientDefs for GetGPUToCPUGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (g_output_[0].IsDense()) {
          return SingleGradientDef(
              "CopyCPUToGPU", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        } else {
          return vector<OperatorDef>{CreateOperatorDef(
                                         "CopyCPUToGPU",
                                         "",
                                         std::vector<string>{GO_I(0)},
                                         std::vector<string>{GI_I(0)}),
                                     CreateOperatorDef(
                                         "CopyCPUToGPU",
                                         "",
                                         std::vector<string>{GO_V(0)},
                                         std::vector<string>{GI_V(0)})};
        }
        */
    }
}
