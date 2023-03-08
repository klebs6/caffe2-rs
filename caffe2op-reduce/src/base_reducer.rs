crate::ix!();

/**
  | Base implementation, everything can
  | be overwritten
  |
  */
pub struct BaseReducer {
    
}

impl Reducer for BaseReducer {
    const InputCount: isize = 1;
}

impl BaseReducer {

    #[inline] pub fn finish<const FixedSize: i32>(&mut self, meta: &BaseReducerMeta, context: *mut CPUContext)  {
    
        todo!();
        /*
        
        */
    }
}
