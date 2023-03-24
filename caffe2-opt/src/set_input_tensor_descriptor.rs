crate::ix!();

#[inline] pub fn set_input_tensor_descriptor_type_and_buffer(
    cpu_tensor: &Tensor, 
    desc:       *mut OnnxTensorDescriptorV1)  
{
    todo!();
    /*
        if (cpu_tensor.template IsType<int32_t>()) {
        desc->dataType = ONNXIFI_DATATYPE_INT32;
        desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int32_t>());
      } else if (cpu_tensor.template IsType<c10::Half>()) {
        desc->dataType = ONNXIFI_DATATYPE_FLOAT16;
        desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<c10::Half>());
      } else if (cpu_tensor.template IsType<float>()) {
        desc->dataType = ONNXIFI_DATATYPE_FLOAT32;
        desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<float>());
      } else if (cpu_tensor.template IsType<int8_t>()) {
        desc->dataType = ONNXIFI_DATATYPE_INT8;
        desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int8_t>());
      } else if (cpu_tensor.template IsType<uint8_t>()) {
        desc->dataType = ONNXIFI_DATATYPE_UINT8;
        desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<uint8_t>());
      } else if (cpu_tensor.template IsType<int64_t>()) {
        desc->dataType = ONNXIFI_DATATYPE_INT64;
        desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int64_t>());
      } else if (cpu_tensor.template IsType<int16_t>()) {
        desc->dataType = ONNXIFI_DATATYPE_INT16;
        desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int16_t>());
      } else if (cpu_tensor.template IsType<uint16_t>()) {
        desc->dataType = ONNXIFI_DATATYPE_UINT16;
        desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<uint16_t>());
      } else {
        CAFFE_THROW(
            "Unsupported tensor type in ONNXIFI: ", cpu_tensor.dtype().name());
      }
    */
}

#[inline] pub fn set_input_tensor_descriptor_type_and_buffer_with_int8tensor_cpu(
    cpu_int8tensor: &Int8TensorCPU, 
    desc:           *mut OnnxTensorDescriptorV1)
{
    todo!();
    /*
        const Tensor& cpu_tensor = cpu_int8tensor.t;
      if (cpu_tensor.template IsType<uint8_t>()) {
        desc->dataType = ONNXIFI_DATATYPE_UINT8;
        desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<uint8_t>());
      } else if (cpu_tensor.template IsType<int8_t>()) {
        desc->dataType = ONNXIFI_DATATYPE_INT8;
        desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int8_t>());
      } else if (cpu_tensor.template IsType<int32_t>()) {
        desc->dataType = ONNXIFI_DATATYPE_INT32;
        desc->buffer = reinterpret_cast<onnxPointer>(cpu_tensor.data<int32_t>());
      } else {
        CAFFE_THROW(
            "Unsupported Int8Tensor type in ONNXIFI: ", cpu_tensor.dtype().name());
      }
      desc->quantizationParams = 1;
      desc->quantizationAxis = 1;
      desc->scales = &cpu_int8tensor.scale;
      desc->biases = &cpu_int8tensor.zero_point;
    */
}
