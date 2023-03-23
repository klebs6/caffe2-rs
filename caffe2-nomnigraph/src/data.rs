crate::ix!();

pub enum DataType { 
    Generic, 
    Float, 
    Half, 
    Int8 
}

pub enum Layout { 
    Generic, 
    NCHW, 
    NHWC 
}

pub struct Data {
    base:    Value,
    version: usize, // default = 0
}

impl Default for Data {
    
    fn default() -> Self {
        todo!();
        /*
            : Value(ValueKind::Data
        */
    }
}

impl Data {
    
    #[inline] pub fn classof(v: *const Value) -> bool {
        
        todo!();
        /*
            return V->getKind() == ValueKind::Data;
        */
    }
    
    #[inline] pub fn get_version(&self) -> usize {
        
        todo!();
        /*
            return version_;
        */
    }
    
    #[inline] pub fn set_version(&mut self, version: usize)  {
        
        todo!();
        /*
            version_ = version;
        */
    }
}

