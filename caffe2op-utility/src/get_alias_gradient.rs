crate::ix!();

pub struct GetAliasGradient;

impl GetGradientDefs for GetAliasGradient {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // We will simply pass-along the gradient. Nothing needs to
        // be calculated.
        SetDense(0, GO(0));
        return vector<OperatorDef>();
        */
    }
}

register_gradient!{Alias, GetAliasGradient}
