crate::ix!();

#[inline] pub fn set_output_tensor_descriptor_type_and_buffer(
    onnxifi_type: u64,
    cpu_tensor:   *mut Tensor,
    desc:         *mut OnnxTensorDescriptorV1)  {
    
    todo!();
    /*
        desc->dataType = onnxifi_type;
      desc->buffer = reinterpret_cast<onnxPointer>(
          cpu_tensor->raw_mutable_data(OnnxifiTypeToDataType(onnxifi_type)));
    */
}
