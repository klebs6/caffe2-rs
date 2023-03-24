crate::ix!();

#[inline] pub fn onnxifi_type_to_data_type(onnxifi_type: u64) -> TypeMeta {
    
    todo!();
    /*
        static std::map<uint64_t, TypeMeta> data_type_map{
          {ONNXIFI_DATATYPE_FLOAT32, TypeMeta::Make<float>()},
          {ONNXIFI_DATATYPE_FLOAT16, TypeMeta::Make<c10::Half>()},
          {ONNXIFI_DATATYPE_INT32, TypeMeta::Make<int>()},
          {ONNXIFI_DATATYPE_INT8, TypeMeta::Make<int8_t>()},
          {ONNXIFI_DATATYPE_UINT8, TypeMeta::Make<uint8_t>()},
          {ONNXIFI_DATATYPE_INT64, TypeMeta::Make<int64_t>()},
          {ONNXIFI_DATATYPE_INT16, TypeMeta::Make<int16_t>()},
          {ONNXIFI_DATATYPE_UINT16, TypeMeta::Make<uint16_t>()},
      };
      const auto it = data_type_map.find(onnxifi_type);
      CAFFE_ENFORCE(
          it != data_type_map.end(),
          "Unsupported ONNXIFI data type: ",
          onnxifi_type);
      return it->second;
    */
}
