crate::ix!();

/**
  | Loads the index from the given 1-D tensor.
  | Elements in the tensor will be given
  | consecutive indexes starting at 1.
  | Fails if tensor contains repeated elements.
  |
  */
pub struct IndexLoadOp {
    storage:          OperatorStorage,
    context:          CPUContext,
    skip_first_entry: bool,
}

num_inputs!{IndexLoad, 2}

num_outputs!{IndexLoad, 1}

inputs!{IndexLoad, 
    0 => ("handle", "Pointer to an Index instance."),
    1 => ("items", "1-D tensor with elements starting with index 1.")
}

outputs!{IndexLoad, 
    0 => ("handle", "The input handle.")
}

args!{IndexLoad, 
    0 => ("skip_first_entry", "If set, skips the first entry of the tensor. 
        This allows to load tensors that are aligned with an embedding, 
        where the first entry corresponds to the default 0 index entry.")
}

scalar_type!{IndexLoad, TensorProto_DataType_UNDEFINED}

enforce_inplace!{IndexLoad, vec![(0, 0)]}

impl IndexLoadOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            skipFirstEntry_( OperatorStorage::GetSingleArgument<int>("skip_first_entry", 0))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<IndexKeyTypes>::call(this, Input(1));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& base = OperatorStorage::Input<std::unique_ptr<IndexBase>>(0);
            auto* dict = dynamic_cast_if_rtti<Index<T>*>(base.get());
            CAFFE_ENFORCE(dict, "Wrong dictionary type given input keys.");
            const auto& keys = Input(1);
            const auto* keys_data = keys.data<T>();
            auto keys_size = keys.numel();
            if (skipFirstEntry_) {
              CAFFE_ENFORCE(keys.numel() > 0);
              ++keys_data;
              --keys_size;
            }
            return dict->Load(keys_data, keys_size);
        */
    }
}
