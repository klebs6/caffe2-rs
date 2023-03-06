crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MapToKeyValueOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

input_tags!{
    MapToKeyValueOp {
        Map
    }
}

output_tags!{
    MapToKeyValueOp {
        Keys,
        Values
    }
}

impl<Context> MapToKeyValueOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<
                    MapType64To64,
                    MapType64To32,
                    MapType32To32,
                    MapType32To64>>::call(this, OperatorStorage::InputBlob(MAP));
        */
    }
    
    #[inline] pub fn do_run_with_type<MAP_T>(&mut self) -> bool {
    
        todo!();
        /*
            using key_type = typename MAP_T::key_type;
                    using mapped_type = typename MAP_T::mapped_type;
                    auto& map_data = this->template Input<MAP_T>(MAP);

                    auto* key_output = Output(
                        KEYS, {static_cast<int64_t>(map_data.size())}, at::dtype<key_type>());
                    auto* value_output = Output(
                        VALUES,
                        {static_cast<int64_t>(map_data.size())},
                        at::dtype<mapped_type>());
                    auto* key_data = key_output->template mutable_data<key_type>();
                    auto* value_data = value_output->template mutable_data<mapped_type>();

                    for (const auto& it : map_data) {
                        *key_data = it.first;
                        *value_data = it.second;
                        key_data++;
                        value_data++;
                    }

                    return true;
        */
    }
}
