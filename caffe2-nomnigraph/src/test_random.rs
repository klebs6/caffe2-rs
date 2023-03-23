crate::ix!();

const A: u32 = 1103515245;
const C: u32 = 12345;

/**
  | Very simple random number generator
  | used to generate platform independent
  | random test data.
  |
  */
pub struct TestRandom {
    seed: u32,
}

impl TestRandom {
    
    pub fn new(seed: u32) -> Self {
    
        todo!();
        /*
            : seed_(seed)
        */
    }
    
    #[inline] pub fn next_int(&mut self) -> u32 {
        
        todo!();
        /*
            seed_ = A * seed_ + C;
        return seed_;
        */
    }
}

