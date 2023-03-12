crate::ix!();

/**
  | Simple Proxy class to expose nicer API
  | for field access
  |
  */
pub struct TreeWalkerField<'a> {
    walker:   &'a TreeWalker<'a>,
    field_id: i32,
}

impl<'a> TreeWalkerField<'a> {
    
    pub fn new(walker: &mut TreeWalker, field_id: i32) -> Self {
        todo!();
        /*
            : walker_(walker), fieldId_(fieldId)
        */
    }
    
    #[inline] pub fn dim(&self) -> Vec<i64> {
        
        todo!();
        /*
            return walker_.fieldDim(fieldId_);
        */
    }
    
    #[inline] pub fn size(&self) -> i64 {
        
        todo!();
        /*
            int64_t size = 1;
          for (const auto d : dim()) {
            size *= d;
          }
          return size;
        */
    }
    
    #[inline] pub fn meta(&self) -> TypeMeta {
        
        todo!();
        /*
            return walker_.input(fieldId_).dtype();
        */
    }
    
    #[inline] pub fn ptr(&self)  {
        
        todo!();
        /*
            return walker_.fieldPtr(fieldId_);
        */
    }
    
    #[inline] pub fn field_id(&self) -> i32 {
        
        todo!();
        /*
            return fieldId_;
        */
    }
    
    #[inline] pub fn offset(&self) -> TOffset {
        
        todo!();
        /*
            return walker_.offset(fieldId_);
        */
    }
}
