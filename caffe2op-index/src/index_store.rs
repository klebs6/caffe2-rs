crate::ix!();

/**
  | Stores the keys of this index in a 1-D
  | tensor. Since element 0 is reserved
  | for unknowns, the first element of the
  | output tensor will be element of index
  | 1.
  |
  */
pub struct IndexStoreOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{IndexStore, 1}

num_outputs!{IndexStore, 1}

inputs!{IndexStore, 
    0 => ("handle", "Pointer to an Index instance.")
}

outputs!{IndexStore, 
    0 => ("items", "1-D tensor with elements starting with index 1.")
}

impl IndexStoreOp {
    
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
        return DispatchHelper<IndexKeyTypes>::call(this, base->Type());
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& base = OperatorStorage::Input<std::unique_ptr<IndexBase>>(0);
            auto* dict = dynamic_cast_if_rtti<Index<T>*>(base.get());
            CAFFE_ENFORCE(dict);
            return dict->Store(Output(0));
        */
    }
}
