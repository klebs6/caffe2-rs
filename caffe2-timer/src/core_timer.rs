crate::ix!();

pub mod timer {
    /*
       typedef std::chrono::high_resolution_clock clock;
       typedef std::chrono::nanoseconds ns;
       */

}

/**
  | @brief
  | 
  | A simple timer object for measuring
  | time.
  | 
  | This is a minimal class around a std::chrono::high_resolution_clock
  | that serves as a utility class for testing
  | code.
  |
  */
pub struct Timer {
    start_time: std::time::Instant,
}

impl Default for Timer {
    
    fn default() -> Self {
        todo!();
        /*
            Start()
        */
    }
}

impl Timer {
    
    /**
      | -----------
      | @brief
      | 
      | Starts a timer.
      |
      */
    #[inline] pub fn start(&mut self)  {
        
        todo!();
        /*
            start_time_ = clock::now();
        */
    }
    
    #[inline] pub fn nano_seconds(&mut self) -> f32 {
        
        todo!();
        /*
            return static_cast<float>(
            std::chrono::duration_cast<ns>(clock::now() - start_time_).count());
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Returns the elapsed time in milliseconds.
      |
      */
    #[inline] pub fn milli_seconds(&mut self) -> f32 {
        
        todo!();
        /*
            return NanoSeconds() / 1000000.f;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Returns the elapsed time in microseconds.
      |
      */
    #[inline] pub fn micro_seconds(&mut self) -> f32 {
        
        todo!();
        /*
            return NanoSeconds() / 1000.f;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Returns the elapsed time in seconds.
      |
      */
    #[inline] pub fn seconds(&mut self) -> f32 {
        
        todo!();
        /*
            return NanoSeconds() / 1000000000.f;
        */
    }
}
