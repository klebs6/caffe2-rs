crate::ix!();

/**
  | Get the current offset in the cursor.
  |
  */
pub struct GetCursorOffsetOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{GetCursorOffset, 1}

num_outputs!{GetCursorOffset, 1}

inputs!{GetCursorOffset, 
    0 => ("cursor", "A blob containing a pointer to the cursor.")
}

outputs!{GetCursorOffset, 
    0 => ("offsets", "Tensor containing the offsets for the cursor.")
}

impl GetCursorOffsetOp {
    
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
        Output(0)->Resize(cursor->offsets.size());
        auto* output = Output(0)->template mutable_data<int>();
        for (size_t i = 0; i < cursor->offsets.size(); ++i) {
          output[i] = cursor->offsets[i];
        }
        return true;
        */
    }
}

