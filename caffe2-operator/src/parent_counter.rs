crate::ix!();

pub struct ParentCounter {
    init_parent_count:  i32,
    parent_count:       Atomic<i32>,
    err_mutex:          parking_lot::RawMutex,
    parent_failed:      AtomicBool,
    err_msg:            String,
}

impl ParentCounter {
    
    pub fn new(init_parent_count: i32) -> Self {
    
        todo!();
        /*
            : init_parent_count_(init_parent_count),
            parent_count(init_parent_count),
            parent_failed(false)
        */
    }
    
    #[inline] pub fn reset(&mut self)  {
        
        todo!();
        /*
            std::unique_lock<std::mutex> lock(err_mutex);
            parent_count = init_parent_count_;
            parent_failed = false;
            err_msg = "";
        */
    }
}

