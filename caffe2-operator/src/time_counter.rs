crate::ix!();

pub struct TimeCounter {
    timer:      Timer,
    start_time: f32, // default = 0.0
    total_time: f32, // default = 0.0
    iterations: i32, // default = 0
}

impl TimeCounter {
    
    #[inline] pub fn average_time(&self) -> f32 {
        
        todo!();
        /*
            return total_time_ / iterations_;
        */
    }
}
