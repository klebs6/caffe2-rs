crate::ix!();

pub struct Counter<T> {
    count: Atomic<T>,
}

impl<T> Counter<T> {
    
    pub fn new(count: T) -> Self {
        todo!();
        /*
            : count_(count)
        */
    }
    
    #[inline] pub fn count_down(&mut self) -> bool {
        
        todo!();
        /*
            if (count_-- > 0) {
          return false;
        }
        return true;
        */
    }
    
    #[inline] pub fn count_up(&mut self) -> T {
        
        todo!();
        /*
            return count_++;
        */
    }
    
    #[inline] pub fn retrieve(&self) -> T {
        
        todo!();
        /*
            return count_.load();
        */
    }
    
    #[inline] pub fn check_if_done(&self) -> T {
        
        todo!();
        /*
            return (count_.load() <= 0);
        */
    }
    
    #[inline] pub fn reset(&mut self, init_count: T) -> T {
        
        todo!();
        /*
            return count_.exchange(init_count);
        */
    }
}
