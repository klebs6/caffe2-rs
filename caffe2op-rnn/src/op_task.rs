crate::ix!();

/**
  | Data structure for a scheduled task
  | in the task queue.
  |
  */
#[derive(Default)]
pub struct OpTask {

    timestep: i32,

    /// matches RNNNetOperator.order
    op_idx: i32,

    /// number of timesteps in this execution
    t: i32,

    /// +1 for forward, -1 for backward pass
    direction: i32,

    /// only used by gpu version
    stream_id: i32, // = -1;
}

impl OpTask {
    
    pub fn new(timestep: i32, op_idx: i32, t: i32, direction: i32) -> Self {
        todo!();
        /*
            : timestep(_timestep), op_idx(_op_idx), T(_T), direction(_direction) 

        CAFFE_ENFORCE(direction == 1 || direction == -1);
        CAFFE_ENFORCE(timestep >= 0 && timestep < _T);
        */
    }
    
    #[inline] pub fn backward(&mut self) -> bool {
        
        todo!();
        /*
            return direction == -1;
        */
    }
    
    #[inline] pub fn forward(&mut self) -> bool {
        
        todo!();
        /*
            return direction == 1;
        */
    }
}
