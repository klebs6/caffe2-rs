crate::ix!();

/**
  | Returns the number of entries currently
  | present in the index.
  |
  */
pub struct IndexSizeOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{IndexSize, 1}

num_outputs!{IndexSize, 1}

inputs!{IndexSize, 
    0 => ("handle", "Pointer to an Index instance.")
}

outputs!{IndexSize, 
    0 => ("items", "Scalar int64 tensor with number of entries.")
}

impl IndexSizeOp {
    
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

        auto* out = Output(0, std::vector<int64_t>{}, at::dtype<int64_tValue>());
        *out->template mutable_data<int64_tValue>() = base->Size();
        return true;
        */
    }
}
