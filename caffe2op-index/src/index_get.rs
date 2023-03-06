crate::ix!();

/**
  | Given an index handle and a tensor of
  | keys, return an Int tensor of same shape
  | containing the indices for each of the
  | keys. If the index is frozen, unknown
  | entries are given index 0. Otherwise,
  | new entries are added into the index.
  | 
  | If an insert is necessary but max_elements
  | has been reached, fail.
  |
  */
pub struct IndexGetOp {
    storage: OperatorStorage,
    context: CPUContext, 
}

num_inputs!{IndexGet, 2}

num_outputs!{IndexGet, 1}

inputs!{IndexGet, 
    0 => ("handle", "Pointer to an Index instance."),
    1 => ("keys", "Tensor of keys to be looked up.")
}

outputs!{IndexGet, 
    0 => ("indices", "Indices for each of the keys.")
}

scalar_type!{IndexGet, TensorProto::INT64}

impl IndexGetOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
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

            auto* values = Output(0, keys.sizes(), at::dtype<int64_tValue>());
            dict->Get(
                keys.data<T>(),
                values->template mutable_data<int64_tValue>(),
                keys.numel());
            return true;
        */
    }
}
