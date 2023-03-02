crate::ix!();

/**
  | Create an unique_ptr blob to hold an
  | atomic<bool>
  |
  */
pub struct CreateAtomicBoolOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{CreateAtomicBool, 0}

num_outputs!{CreateAtomicBool, 1}

outputs!{CreateAtomicBool, 
    0 => ("atomic_bool", "Blob containing a unique_ptr<atomic<bool>>")
}

should_not_do_gradient!{CreateAtomicBool}

register_cpu_operator!{CreateAtomicBool, CreateAtomicBoolOp}

impl CreateAtomicBoolOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<std::unique_ptr<std::atomic<bool>>>(0) =
            std::unique_ptr<std::atomic<bool>>(new std::atomic<bool>(false));
        return true;
        */
    }
}

