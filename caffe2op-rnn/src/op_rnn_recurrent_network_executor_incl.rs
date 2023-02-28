crate::ix!();

use crate::{
    OperatorDef,
    OperatorStorage,
};

/**
  | Struct for operator in a timestep and
  | its dependencies.
  |
  */
pub struct RNNNetOperator {

    /**
      | Position in the step net (i.e nth operator)
      |
      */
    order:                   i32,

    op:                      Arc<OperatorStorage>,

    /**
      | Special flag for link op, see
      | RNNApplyLinkOp.
      |
      */
    link_op:                 bool,

    /**
      | Bookkeeping, used by
      | ThreadedRecurrentNetworkExecutor
      |
      */
    num_dynamic_inputs:      i32, // = 0;

    num_recurrent_inputs:    i32, // = 0;

    proc_inputs:             Atomic<i32>,

    /**
      | Dependencies to other ops. If dependency
      | index < order, it is a recurrent dependency
      | (i.e to the next timestep)
      |
      */
    dependencies:            Vec<i32>,

    parents:                 Vec<i32>,

    /// For ops that are launched first
    frontier:                bool, // = true; 

    has_timestep_blob:       bool, // = false;
}

impl RNNNetOperator {
    
    pub fn new(def: &OperatorDef, order: i32) -> Self {
        todo!();
        /*
            : order(order) 

        proc_inputs = 0;
        link_op = def.type() == "rnn_internal_apply_link";
        */
    }
    
    pub fn new_from_other(x: &RNNNetOperator) -> Self {
        todo!();
        /*
            order = x.order;
        op = x.op;
        link_op = x.link_op;
        num_dynamic_inputs = x.num_dynamic_inputs;
        num_recurrent_inputs = x.num_recurrent_inputs;
        proc_inputs = 0;
        dependencies = x.dependencies;
        parents = x.parents;
        frontier = x.frontier;
        */
    }
}

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
