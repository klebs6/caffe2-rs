crate::ix!();

pub struct GetUpsampleBilinearGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetUpsampleBilinearGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (def_.input().size() == 2) {
          // this is a hack to support the second input as dynamic
          // width_scale and height_scale to align with onnx change
          return SingleGradientDef(
              "UpsampleBilinearGradient",
              "",
              vector<string>{GO(0), I(0), I(1)},
              vector<string>{GI(0)});
        }
        return SingleGradientDef(
            "UpsampleBilinearGradient",
            "",
            vector<string>{GO(0), I(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    UpsampleBilinear, 
    GetUpsampleBilinearGradient
}
