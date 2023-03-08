crate::ix!();

pub struct GetReshapeGradient {

}

impl GetGradientDefs for GetReshapeGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Reshape",
            "",
            vector<string>{GO(0), O(1)},
            vector<string>{GI(0), "_" + GI(0) + "_dims"});
        */
    }
}

impl CopyArguments for GetReshapeGradient {

    /**
      | Argument `shape` is no longer needed
      | in backprop.
      |
      */
    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

register_gradient!{Reshape, GetReshapeGradient}
