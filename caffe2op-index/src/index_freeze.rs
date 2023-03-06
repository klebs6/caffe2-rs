crate::ix!();

/**
  | Freezes the given index, disallowing
  | creation of new index entries.
  | 
  | Should not be called concurrently with
  | IndexGet.
  |
  */
pub struct IndexFreezeOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{IndexFreeze, 1}

num_outputs!{IndexFreeze, 1}

inputs!{IndexFreeze, 
    0 => ("handle", "Pointer to an Index instance.")
}

outputs!{IndexFreeze, 
    0 => ("handle", "The input handle.")
}

scalar_type!{IndexFreeze, TensorProto_DataType_UNDEFINED}

enforce_inplace!{IndexFreeze, vec![(0, 0)]}

impl IndexFreezeOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& base = OperatorStorage::Input<std::unique_ptr<IndexBase>>(0);
        base->Freeze();
        return true;
        */
    }
}
