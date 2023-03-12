crate::ix!();

/**
  | @brief
  | 
  | CounterSerializer is the serializer
  | for Counter type.
  | 
  | CounterSerializer takes in a blob that
  | contains a Counter, and serializes
  | it into a BlobProto protocol buffer.
  | At the moment only int64_t counters
  | are supported (since it's the only once
  | that is really used).
  |
  */
pub struct CounterSerializer {
    base: dyn BlobSerializerBase,
}

impl CounterSerializer {
    
    #[inline] pub fn serialize(
        &mut self, 
        pointer:       *const c_void,
        type_meta:     TypeMeta,
        name:          &String,
        acceptor:      SerializationAcceptor)  
    {
        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<std::unique_ptr<Counter<int64_t>>>());

        BlobProto blob_proto;
        blob_proto.set_name(name);
        blob_proto.set_type("std::unique_ptr<Counter<int64_t>>");
        TensorProto& proto = *blob_proto.mutable_tensor();
        proto.set_name(name);
        proto.set_data_type(TensorProto_DataType_INT64);
        proto.add_dims(1);
        proto.add_int64_data(
            (*static_cast<const std::unique_ptr<Counter<int64_t>>*>(pointer))
                ->retrieve());
        acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}
