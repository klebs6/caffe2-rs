crate::ix!();

pub struct IndexBase {
    max_elements:  i64,
    meta:          TypeMeta,
    next_id:       i64, //{1}; // guarded by dictMutex_
    frozen_:       AtomicBool, //{false};
    dict_mutex:    parking_lot::RawMutex,
}

impl IndexBase {
    
    pub fn new(max_elements: i64, ty: TypeMeta) -> Self {
        todo!();
        /*
            : maxElements_{maxElements}, meta_(type), frozen_{false}
        */
    }
    
    #[inline] pub fn freeze(&mut self)  {
        
        todo!();
        /*
            frozen_ = true;
        */
    }
    
    #[inline] pub fn is_frozen(&self) -> bool {
        
        todo!();
        /*
            return frozen_;
        */
    }
    
    #[inline] pub fn max_elements(&self) -> i64 {
        
        todo!();
        /*
            return maxElements_;
        */
    }
    
    #[inline] pub fn typemeta(&self) -> TypeMeta {
        
        todo!();
        /*
            return meta_;
        */
    }
    
    #[inline] pub fn size(&mut self) -> i64 {
        
        todo!();
        /*
            std::lock_guard<std::mutex> guard(dictMutex_);
        return nextId_;
        */
    }
}
