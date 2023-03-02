crate::ix!();

/**
  | Set an atomic<bool> to true if the given
  | condition bool variable is true
  |
  */
pub struct ConditionalSetAtomicBoolOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{ConditionalSetAtomicBool, 2}

num_outputs!{ConditionalSetAtomicBool, 0}

inputs!{ConditionalSetAtomicBool, 
    0 => ("atomic_bool", "Blob containing a unique_ptr<atomic<bool>>"),
    1 => ("condition", "Blob containing a bool")
}

should_not_do_gradient!{ConditionalSetAtomicBool}

register_cpu_operator!{ConditionalSetAtomicBool, ConditionalSetAtomicBoolOp}

impl ConditionalSetAtomicBoolOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& ptr =
            OperatorStorage::Input<std::unique_ptr<std::atomic<bool>>>(ATOMIC_BOOL);
        if (Input(CONDITION).data<bool>()[0]) {
          ptr->store(true);
        }
        return true;
        */
    }
}

input_tags!{
    ConditionalSetAtomicBoolOp {
        AtomicBool,
        Condition
    }
}

