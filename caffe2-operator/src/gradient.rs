crate::ix!();

/**
  | -----------
  | @brief
  | 
  | A helper class to indicate that the operator
  | does not need gradient computation.
  | 
  | Use the macro NO_GRADIENT to register
  | operators that do not have gradients.
  | 
  | -----------
  | @note
  | 
  | this is different fron SHOULD_NOT_DO_GRADIENT:
  | the latter means that the gradient computation
  | should not flow through it at all, and
  | throws an error if it is called.
  |
  */
pub struct oGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for oGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return vector<OperatorDef>();
        */
    }
}
