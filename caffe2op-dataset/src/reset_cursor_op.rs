crate::ix!();

/**
  | Resets the offsets for the given TreeCursor.
  | This operation is thread safe.
  |
  */
pub struct ResetCursorOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{ResetCursor, 1}

num_outputs!{ResetCursor, 0}

inputs!{ResetCursor, 
    0 => ("cursor", "A blob containing a pointer to the cursor.")
}

impl ResetCursorOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& cursor = OperatorStorage::Input<std::unique_ptr<TreeCursor>>(0);
        std::lock_guard<std::mutex> lock(cursor->mutex_);
        cursor->offsets.clear();
        return true;
        */
    }
}

