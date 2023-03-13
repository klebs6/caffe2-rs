crate::ix!();

/**
  | Creates a dictionary that maps T keys
  | to consecutive integers from 1 to max_elements.
  | Zero is reserved for unknown keys.
  | 
  | TODO(azzolini): support sizes larger
  | than int32
  |
  */
pub struct IndexCreateOp<T> {
    storage:      OperatorStorage,
    context:      CPUContext,
    max_elements: i64,
    phantom:      PhantomData<T>,
}

num_inputs!{IndexCreate, 0}

num_outputs!{IndexCreate, 1}

outputs!{IndexCreate, 
    0 => ("handle", "Pointer to an Index instance.")
}

args!{IndexCreate, 
    0 => ("max_elements", "Max number of elements, including the zero entry.")
}

scalar_type!{IndexCreate, TensorProto_DataType_UNDEFINED}

impl<T> IndexCreateOp<T> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            maxElements_(OperatorStorage::GetSingleArgument<int>( "max_elements", int::max))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<std::unique_ptr<IndexBase>>(0) =
            std::unique_ptr<IndexBase>(new Index<T>(maxElements_));
        return true;
        */
    }
}
