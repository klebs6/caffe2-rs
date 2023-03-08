crate::ix!();

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
