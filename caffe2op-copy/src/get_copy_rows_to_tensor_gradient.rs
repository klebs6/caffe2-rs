crate::ix!();

pub struct GetCopyRowsToTensorGradient {}

impl GetGradientDefs for GetCopyRowsToTensorGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (g_output_[0].IsDense()) {
          return SingleGradientDef(
              "CopyRowsToTensorGradient",
              "",
              vector<string>{GO(0)},
              vector<string>{GI(0)});
        } else {
          return vector<OperatorDef>{CreateOperatorDef(
                                         "CopyRowsToTensorGradient",
                                         "",
                                         std::vector<string>{GO_I(0)},
                                         std::vector<string>{GI_I(0)}),
                                     CreateOperatorDef(
                                         "CopyRowsToTensorGradient",
                                         "",
                                         std::vector<string>{GO_V(0)},
                                         std::vector<string>{GI_V(0)})};
        }
        */
    }
}

register_gradient!{
    CopyRowsToTensor, 
    GetCopyRowsToTensorGradient
}
