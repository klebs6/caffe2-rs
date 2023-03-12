crate::ix!();

pub struct GetCPUToGPUGradient;

impl GetGradientDefs for GetCPUToGPUGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (g_output_[0].IsDense()) {
          return SingleGradientDef(
              "CopyGPUToCPU", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        } else {
          return vector<OperatorDef>{CreateOperatorDef(
                                         "CopyGPUToCPU",
                                         "",
                                         std::vector<string>{GO_I(0)},
                                         std::vector<string>{GI_I(0)}),
                                     CreateOperatorDef(
                                         "CopyGPUToCPU",
                                         "",
                                         std::vector<string>{GO_V(0)},
                                         std::vector<string>{GI_V(0)})};
        }
        */
    }
}

register_gradient!{Copy,         GetCopyGradient}
register_gradient!{CopyGPUToCPU, GetGPUToCPUGradient}
register_gradient!{CopyCPUToGPU, GetCPUToGPUGradient}
