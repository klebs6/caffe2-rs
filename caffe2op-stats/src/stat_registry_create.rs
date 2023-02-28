crate::ix!();

/**
  | Create a StatRegistry object that will
  | contain a map of performance counters
  | keyed by name.
  | 
  | A StatRegistry is used to gather and
  | retrieve performance counts throughout
  | the caffe2 codebase.
  |
  */
pub struct StatRegistryCreateOp {
    storage: OperatorStorage,
    context: CPUContext,
}

register_cpu_operator!{StatRegistryCreate, StatRegistryCreateOp}

num_inputs!{StatRegistryCreate, 0}

num_outputs!{StatRegistryCreate, 1}

outputs!{StatRegistryCreate, 
    0 => ("handle", "A Blob pointing to the newly created StatRegistry.")
}

impl StatRegistryCreateOp {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<std::unique_ptr<StatRegistry>>(0) =
            std::unique_ptr<StatRegistry>(new StatRegistry);
        return true;
        */
    }
}


caffe_known_type!{Box<StatRegistry>}
