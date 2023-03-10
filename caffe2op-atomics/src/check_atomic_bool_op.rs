crate::ix!();

/**
  | Copy the value of an atomic<bool> to
  | a bool
  |
  */
pub struct CheckAtomicBoolOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{CheckAtomicBool, 1}

num_outputs!{CheckAtomicBool, 1}

inputs!{CheckAtomicBool, 
    0 => ("atomic_bool", "Blob containing a unique_ptr<atomic<bool>>")
}

outputs!{CheckAtomicBool, 
    0 => ("value", "Copy of the value for the atomic<bool>")
}

should_not_do_gradient!{CheckAtomicBool}

register_cpu_operator!{CheckAtomicBool, CheckAtomicBoolOp}

impl CheckAtomicBoolOp {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& ptr = OperatorStorage::Input<std::unique_ptr<std::atomic<bool>>>(0);
        Output(0)->Resize(1);
        *Output(0)->template mutable_data<bool>() = ptr->load();
        return true;
        */
    }
}
