crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct KeyValueToMapOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

input_tags!{
    KeyValueToMapOp {
        Keys,
        Values
    }
}

output_tags!{
    KeyValueToMapOp {
        Map
    }
}

impl<Context> KeyValueToMapOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
                    this, Input(KEYS));
        */
    }
    
    #[inline] pub fn do_run_with_type<KEY_T>(&mut self) -> bool {
    
        todo!();
        /*
            return DispatchHelper<
                        TensorTypes2<int32_t, int64_t, GenericTensorImplementation>,
                        KEY_T>::call(this, Input(VALUES));
        */
    }
    
    #[inline] pub fn do_run_with_type2<KEY_T, VALUE_T>(&mut self) -> bool {
    
        todo!();
        /*
            using MapType = typename MapTypeTraits<KEY_T, VALUE_T>::MapType;
                    const auto& key_input = Input(KEYS);
                    const auto& value_input = Input(VALUES);

                    CAFFE_ENFORCE_EQ(key_input.numel(), value_input.numel());

                    auto* key_data = key_input.template data<KEY_T>();
                    auto* value_data = value_input.template data<VALUE_T>();

                    auto* map_data = this->template Output<MapType>(MAP);

                    for (int i = 0; i < key_input.numel(); ++i) {
                        map_data->emplace(key_data[i], value_data[i]);
                    }

                    return true;
        */
    }
    
    #[inline] pub fn do_run_with_other_type2<KEY_T>(&mut self) -> bool {
    
        todo!();
        /*
            CAFFE_THROW(
                        "KeyValueToMap is not implemented on value tensor of type ",
                        Input(VALUES).dtype().name(),
                        "consider adding it as a type in the DispatchHelper list");
        */
    }
}
