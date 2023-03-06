crate::ix!();

pub type MapType64To64 = HashMap<i64,i64>;
pub type MapType64To32 = HashMap<i64,i32>;
pub type MapType32To32 = HashMap<i32,i32>;
pub type MapType32To64 = HashMap<i32,i64>;

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
