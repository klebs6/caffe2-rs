crate::ix!();

#[inline] pub fn onnxifi_data_type(t: TensorProto_DataType) -> u64 {
    
    todo!();
    /*
        #define CAFFE2_TO_ONNXIFI_TYPE(x, y) \
      case (TensorProto::x):     \
        return y
      switch (t) {
        CAFFE2_TO_ONNXIFI_TYPE(FLOAT, ONNXIFI_DATATYPE_FLOAT32);
        CAFFE2_TO_ONNXIFI_TYPE(INT8, ONNXIFI_DATATYPE_INT8);
        CAFFE2_TO_ONNXIFI_TYPE(UINT8, ONNXIFI_DATATYPE_UINT8);
        CAFFE2_TO_ONNXIFI_TYPE(INT16, ONNXIFI_DATATYPE_INT16);
        CAFFE2_TO_ONNXIFI_TYPE(UINT16, ONNXIFI_DATATYPE_UINT16);
        CAFFE2_TO_ONNXIFI_TYPE(INT32, ONNXIFI_DATATYPE_INT32);
        CAFFE2_TO_ONNXIFI_TYPE(INT64, ONNXIFI_DATATYPE_INT64);
        CAFFE2_TO_ONNXIFI_TYPE(FLOAT16, ONNXIFI_DATATYPE_FLOAT16);
        default:
          LOG(WARNING) << "Unsupported Caffe2 tensor type: " << t
                       << ", fallback to FLOAT";
          return ONNXIFI_DATATYPE_FLOAT32;
      }
    #undef CAFFE2_TO_ONNXIFI_TYPE
    */
}
