crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CreateMapOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

output_tags!{
    CreateMapOp {
        Map
    }
}

impl<Context> CreateMapOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            TensorProto::DataType key_dtype = static_cast<TensorProto::DataType>(
                    this->template GetSingleArgument<int>(
                        "key_dtype", TensorProto_DataType_INT32));

                return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
                    this, DataTypeToTypeMeta(key_dtype));
        */
    }
    
    #[inline] pub fn do_run_with_type<KEY_T>(&mut self) -> bool {
    
        todo!();
        /*
            TensorProto::DataType value_dtype = static_cast<TensorProto::DataType>(
                        this->template GetSingleArgument<int>(
                            "value_dtype", TensorProto_DataType_INT32));

                    return DispatchHelper<
                        TensorTypes2<int32_t, int64_t, GenericTensorImplementation>,
                        KEY_T>::call(this, DataTypeToTypeMeta(value_dtype));
        */
    }
    
    #[inline] pub fn do_run_with_type2<KEY_T, VALUE_T>(&mut self) -> bool {
    
        todo!();
        /*
            // clear to make sure the map is empty
                    this->template Output<typename MapTypeTraits<KEY_T, VALUE_T>::MapType>(MAP)
                        ->clear();
                    return true;
        */
    }
    
    #[inline] pub fn do_run_with_other_type2<KEY_T>(&mut self) -> bool {
    
        todo!();
        /*
            TensorProto::DataType value_dtype = static_cast<TensorProto::DataType>(
                        this->template GetSingleArgument<int>(
                            "value_dtype", TensorProto_DataType_INT32));

                    CAFFE_THROW(
                        "CreateMap is not implemented on value tensor of type ",
                        DataTypeToTypeMeta(value_dtype).name(),
                        "consider adding it as a type in the DispatchHelper list");
        */
    }
}
