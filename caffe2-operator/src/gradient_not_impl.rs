crate::ix!();

/**
  | -----------
  | @brief
  | 
  | A helper class to indicate that the gradient
  | mechanism is not ready.
  | 
  | This should only be used sparsely when
  | the gradient does exist, but we have
  | not implemented it yet and are using
  | this as a lazy excuse. Eventually, a
  | gradient operator should be implemented.
  |
  */
pub struct GradientNotImplementedYet<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GradientNotImplementedYet<'a> {
    
    #[inline] pub fn get(&mut self) -> GradientOpsMeta {
        
        todo!();
        /*
            CAFFE_THROW(
            "Operator ",
            def_.type(),
            " should have a gradient but is not implemented yet.");
        */
    }
}


