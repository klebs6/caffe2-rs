crate::ix!();

#[inline] pub fn get_onnxifi_data_type(t: TensorProto_DataType) -> u64 {
    
    todo!();
    /*
        #define CAFFE2_TO_ONNXIFI_TYPE(x) \
      case (caffe2::TensorProto::x):  \
        return ONNXIFI_DATATYPE_##x
      switch (t) {
        CAFFE2_TO_ONNXIFI_TYPE(INT8);
        CAFFE2_TO_ONNXIFI_TYPE(UINT8);
        CAFFE2_TO_ONNXIFI_TYPE(UINT16);
        CAFFE2_TO_ONNXIFI_TYPE(INT16);
        CAFFE2_TO_ONNXIFI_TYPE(INT32);
        CAFFE2_TO_ONNXIFI_TYPE(INT64);
        CAFFE2_TO_ONNXIFI_TYPE(FLOAT16);
        case (caffe2::TensorProto::FLOAT):
          return ONNXIFI_DATATYPE_FLOAT32;
        default:
          LOG(WARNING) << "Unsupported Caffe2 tensor type: " << t;
          return ONNXIFI_DATATYPE_UNDEFINED;
      }
    #undef CAFFE2_TO_ONNXIFI_TYPE
    */
}

