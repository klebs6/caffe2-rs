crate::ix!();

/**
  | @brief
  | 
  | CounterDeserializer is the deserializer
  | for Counters.
  |
  */
pub struct CounterDeserializer {
    base: dyn BlobDeserializerBase,
}

impl CounterDeserializer {
    
    #[inline] pub fn deserialize(&mut self, proto: &BlobProto, blob: *mut Blob)  {
        
        todo!();
        /*
            auto tensorProto = proto.tensor();
        CAFFE_ENFORCE_EQ(tensorProto.dims_size(), 1, "Unexpected size of dims");
        CAFFE_ENFORCE_EQ(tensorProto.dims(0), 1, "Unexpected value of dims");
        CAFFE_ENFORCE_EQ(
            tensorProto.data_type(),
            TensorProto_DataType_INT64,
            "Only int64_t counters supported");
        CAFFE_ENFORCE_EQ(
            tensorProto.int64_data_size(), 1, "Unexpected size of data");
        *blob->GetMutable<std::unique_ptr<Counter<int64_t>>>() =
            std::make_unique<Counter<int64_t>>(tensorProto.int64_data(0));
        */
    }
}
