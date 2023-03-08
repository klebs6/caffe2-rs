crate::ix!();

use crate::{
    OperatorStorage,
};

/**
  | A helper class to denote that an op does
  | not have a default engine.
  | 
  | NoDefaultEngineOp is a helper class
  | that one can use to denote that a specific
  | operator is not intended to be called
  | without an explicit engine given. This
  | is the case for e.g. the communication
  | operators where one has to specify a
  | backend (like MPI or ZEROMQ).
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct NoDefaultEngineOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

impl<Context> NoDefaultEngineOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_THROW(
            "The operator ",
            this->debug_def().type(),
            " does not have a default engine implementation. Please "
            "specify an engine explicitly for this operator.");
        */
    }
}
