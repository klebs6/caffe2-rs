crate::ix!();

pub struct GetLayerNormGradient;

impl GetGradientDefs for GetLayerNormGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            bool elementwise_affine = false;
            if (ArgumentHelper::HasArgument(Def(), "elementwise_affine")) {
              elementwise_affine = GetArgument(Def(), "elementwise_affine").i();
            }
            if (elementwise_affine) {
              return SingleGradientDef(
                  "LayerNormGradient",
                  "",
                  std::vector<std::string>{GO(0), O(0), O(1), O(2), I(0), I(1)},
                  std::vector<std::string>{GI(0), GI(1), GI(2)});
            } else {
              return SingleGradientDef(
                  "LayerNormGradient",
                  "",
                  std::vector<std::string>{GO(0), O(0), O(1), O(2), I(0)},
                  std::vector<std::string>{GI(0)});
            }
        */
    }
}

register_gradient!{
    LayerNorm, 
    GetLayerNormGradient
}
