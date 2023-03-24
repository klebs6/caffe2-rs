crate::ix!();

pub struct SchemaArgument {
    name:        *const u8,
    description: *const u8,
    required:    bool,
}

impl SchemaArgument {
    
    pub fn new(
        name:        *const u8,
        description: *const u8,
        required:    bool) -> Self {
    
        todo!();
        /*
            : name_{name}, description_{description}, required_{required}
        */
    }
    
    #[inline] pub fn name(&self) -> *const u8 {
        
        todo!();
        /*
            return name_;
        */
    }
    
    #[inline] pub fn description(&self) -> *const u8 {
        
        todo!();
        /*
            return description_;
        */
    }
    
    #[inline] pub fn is_required(&self) -> bool {
        
        todo!();
        /*
            return required_;
        */
    }
}
