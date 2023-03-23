crate::ix!();

/**
  | Implements accessors for a generic type T. If
  | the type is not specified (i.e., void template
  | type) then the partial specification gives an
  | empty type.
  */
pub struct StorageType<T> {
    data: T,
}

impl<T> StorageType<T> {
    
    pub fn new(data: T) -> Self {
    
        todo!();
        /*
            : Data(std::move(data))
        */
    }
    
    #[inline] pub fn data(&self) -> &T {
        
        todo!();
        /*
            return Data;
        */
    }
    
    #[inline] pub fn mutable_data(&mut self) -> *mut T {
        
        todo!();
        /*
            return &Data;
        */
    }
    
    #[inline] pub fn reset_data(&mut self, data: T)  {
        
        todo!();
        /*
            Data = std::move(data);
        */
    }
}


