crate::ix!();

pub struct GetLRNGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLRNGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
          "LRNGradient", "",
          vector<string>{I(0), O(0), GO(0)},
          vector<string>{GI(0)});
        */
    }
}

register_gradient!{LRN, GetLRNGradient}

