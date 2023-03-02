crate::ix!();

/**
  | Creates an unlocked mutex and returns
  | it in a unique_ptr blob.
  |
  */
pub struct CreateMutexOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{CreateMutex, 0}

num_outputs!{CreateMutex, 1}

outputs!{CreateMutex, 
    0 => ("mutex_ptr", "Blob containing a std::unique_ptr<mutex>.")
}

scalar_type!{
    CreateMutex, 
    TensorProto_DataType_UNDEFINED
}

should_not_do_gradient!{CreateMutex}

register_cpu_operator!{
    CreateMutex, 
    CreateMutexOp
}

register_ideep_operator!{
    CreateMutex, 
    IDEEPFallbackOp::<CreateMutexOp, SkipIndices<0>>
}

impl<Context> CreateMutexOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<std::unique_ptr<std::mutex>>(0) =
            std::unique_ptr<std::mutex>(new std::mutex);
        return true;
        */
    }
}

