crate::ix!();

use crate::{
    BlobSerializationOptions,
    BlobSerializerBase,
    OperatorStorage,
    Blob,
    TypeMeta,
    BlobProto,
    SerializationAcceptor
};

pub type MapType64To64 = HashMap<i64,i64>;
pub type MapType64To32 = HashMap<i64,i32>;
pub type MapType32To32 = HashMap<i32,i32>;
pub type MapType32To64 = HashMap<i32,i64>;

///------------------
pub struct CreateMapOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS

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

///-------------------------------
pub struct KeyValueToMapOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
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

///----------------------------
pub struct MapToKeyValueOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
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

pub struct MapSerializer<KEY_T,VALUE_T> {

    /**
      | using MapType = typename MapTypeTraits<KEY_T,
      | VALUE_T>::MapType;
      |
      */
    phantomKEY_T: PhantomData<KEY_T>,
    phantomVALUE_T: PhantomData<VALUE_T>,
}

impl<K,V> BlobSerializerBase for MapSerializer<K,V> {

    #[inline] fn serialize(&mut self, 
        pointer:   *const libc::c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor,
        options:   Option<&BlobSerializationOptions>)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<MapType>());
                const MapType& map_data = *static_cast<const MapType*>(pointer);
                int64_t sz = map_data.size();
                Tensor key_tensor(CPU);
                key_tensor.Resize(sz);
                Tensor value_tensor(CPU);
                value_tensor.Resize(sz);
                auto* key_data = key_tensor.mutable_data<KEY_T>();
                auto* value_data = value_tensor.mutable_data<VALUE_T>();
                for (const auto& it : map_data) {
                    *key_data = it.first;
                    *value_data = it.second;
                    key_data++;
                    value_data++;
                }

                TensorProtos tensor_protos;
                TensorSerializer ser;
                ser.Serialize(
                    key_tensor, name, tensor_protos.add_protos(), 0, key_tensor.numel());
                ser.Serialize(
                    value_tensor,
                    name,
                    tensor_protos.add_protos(),
                    0,
                    value_tensor.numel());

                BlobProto blob_proto;
                blob_proto.set_name(name);
                blob_proto.set_type(MapTypeTraits<KEY_T, VALUE_T>::MapTypeName());
                blob_proto.set_content(SerializeAsString_EnforceCheck(tensor_protos));
                acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}

///-----------------------------------
pub struct MapDeserializer<KEY_T,VALUE_T> {
    //using MapType = typename MapTypeTraits<KEY_T, VALUE_T>::MapType;
    phantomKEY_T: PhantomData<KEY_T>,
    phantomVALUE_T: PhantomData<VALUE_T>,
}

impl<K,V> BlobDeserializerBase for MapDeserializer<K,V> {

    #[inline] fn deserialize(&mut self, proto: &BlobProto, blob: *mut Blob)  {
        
        todo!();
        /*
            TensorProtos tensor_protos;
                CAFFE_ENFORCE(
                    tensor_protos.ParseFromString(proto.content()),
                    "Fail to parse TensorProtos");
                TensorDeserializer deser;
                Tensor key_tensor = deser.Deserialize(tensor_protos.protos(0));
                Tensor value_tensor = deser.Deserialize(tensor_protos.protos(1));
                auto* key_data = key_tensor.data<KEY_T>();
                auto* value_data = value_tensor.data<VALUE_T>();

                auto* map_ptr = blob->template GetMutable<MapType>();
                for (int i = 0; i < key_tensor.numel(); ++i) {
                    map_ptr->emplace(key_data[i], value_data[i]);
                }
        */
    }
}

caffe_known_type!{MapType64To64}
caffe_known_type!{MapType64To32}
caffe_known_type!{MapType32To32}
caffe_known_type!{MapType32To64}

register_blob_serializer!{
    /*
       TypeMeta::Id<MapType64To64>(),
       MapSerializer<int64_t, int64_t>
       */
}

register_blob_serializer!{
    /*
       TypeMeta::Id<MapType64To32>(),
       MapSerializer<int64_t, int32_t>
       */
}

register_blob_serializer!{
    /*
       TypeMeta::Id<MapType32To32>(),
       MapSerializer<int32_t, int32_t>
       */
}

register_blob_serializer!{
    /*
       TypeMeta::Id<MapType32To64>(),
       MapSerializer<int32_t, int64_t>
       */
}

register_blob_deserializer!{
    /*
       (std::unordered_map<int64_t, int64_t>),
       MapDeserializer<int64_t, int64_t>
       */
}

register_blob_deserializer!{
    /*
       (std::unordered_map<int64_t, int32_t>),
       MapDeserializer<int64_t, int32_t>
       */
}

register_blob_deserializer!{
    /*
       (std::unordered_map<int32_t, int32_t>),
       MapDeserializer<int32_t, int32_t>
       */
}

register_blob_deserializer!{
    /*
       (std::unordered_map<int32_t, int64_t>),
       MapDeserializer<int32_t, int64_t>
*/
}

/**
  | Create an empty map blob
  |
  */
register_cpu_operator!{CreateMap, CreateMapOp<CPUContext>}

num_inputs!{CreateMap, 0}

num_outputs!{CreateMap, 1}

outputs!{CreateMap, 
    0 => ("map blob", "Blob reference to the map")
}

args!{CreateMap, 
    0 => ("key_dtype", "Key's TensorProto::DataType (default INT32)"),
    1 => ("value_dtype", "Value's TensorProto::DataType (default INT32)")
}

scalar_type!{CreateMap, TensorProto_DataType_UNDEFINED}

/**
  | Convert key and value blob pairs into
  | a map blob
  |
  */
register_cpu_operator!{KeyValueToMap, KeyValueToMapOp<CPUContext>}

num_inputs!{KeyValueToMap, 2}

num_outputs!{KeyValueToMap, 1}

inputs!{KeyValueToMap, 
    0 => ("key blob", "Blob reference to the key"),
    1 => ("value blob", "Blob reference to the value")
}

outputs!{KeyValueToMap, 
    0 => ("map blob", "Blob reference to the map")
}

/**
  | Convert a map blob into key and value
  | blob pairs
  |
  */
register_cpu_operator!{MapToKeyValue, MapToKeyValueOp<CPUContext>}

num_inputs!{MapToKeyValue, 1}

num_outputs!{MapToKeyValue, 2}

inputs!{MapToKeyValue, 
    0 => ("map blob", "Blob reference to the map")
}

outputs!{MapToKeyValue, 
    0 => ("key blob", "Blob reference to the key"),
    1 => ("value blob", "Blob reference to the value")
}
