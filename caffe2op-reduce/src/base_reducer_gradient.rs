crate::ix!();

pub struct BaseReducerGradient {
    
}

impl BaseReducerGradient {

    /// which of the original inputs are required for gradient computation
    #[inline] pub fn original_inputs() -> [i32; 0] {
        todo!();
        /*
           return std::array<int, 0>();
        */
    }
    
    #[inline] pub fn compute_length() -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    #[inline] pub fn num_aux_inputs_with_grads(def: &OperatorDef) -> i32 {
        
        todo!();
        /*
            return 0;
        */
    }
    
    #[inline] pub fn requires_data_input(def: &OperatorDef) -> bool {
        
        todo!();
        /*
            return false;
        */
    }

    /// True if the backward op requires the output of the forward op.
    #[inline] pub fn requires_forward_output() -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}


