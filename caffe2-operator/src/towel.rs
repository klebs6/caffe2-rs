crate::ix!();

/**
  | -----------
  | @brief
  | 
  | A helper class to indicate that the operator
  | should have no gradient.
  | 
  | This is used when the operator definition
  | is designed to not have a gradient.
  | 
  | Calling a gradient on this operator
  | def will cause Caffe2 to quit.
  |
  */
pub struct ThrowInTheTowelIfGradientIsCalled<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> ThrowInTheTowelIfGradientIsCalled<'a> {
    
    #[inline] pub fn get(&mut self) -> GradientOpsMeta {
        
        todo!();
        /*
            CAFFE_THROW("One should not call gradient for operator ", def_.type(), ".");
        */
    }
}


