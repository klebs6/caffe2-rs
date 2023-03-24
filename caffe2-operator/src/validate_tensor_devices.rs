crate::ix!();

#[inline] pub fn validate_tensor_devices<Context>(
    op:     &mut OperatorStorage,
    op_def: &OperatorDef) -> HashMap<String,(DeviceOption,DeviceOption)> 
{
    todo!();
    /*
        std::map<string, std::pair<DeviceOption, DeviceOption>> mismatches;
      DeviceOption op_device = op_def.device_option();

    #ifndef CAFFE2_NO_OPERATOR_SCHEMA
      // Check from op schema if this op is used for crossing devices
      auto op_schema = OpSchemaRegistry::Schema(op_def.type());
      if (op_schema != nullptr) {
        if (op_schema->inputs_can_cross_devices()) {
          return mismatches;
        }
      }
    #endif // CAFFE2_NO_OPERATOR_SCHEMA

      auto Check = [&](const Blob& blob, std::string blob_name) {
        TensorInfoCall tensor_info_fun = GetTensorInfoFunction(blob.meta().id());
        if (tensor_info_fun) {
          size_t _capacity;
          DeviceOption blob_device;
          tensor_info_fun(
              const_cast<Blob&>(blob).GetRaw(), &_capacity, &blob_device);

          if ((blob_device.device_type() == PROTO_CUDA ||
               blob_device.device_type() == PROTO_HIP) &&
              blob_device.device_id() != op_device.device_id()) {
            mismatches[blob_name] = std::make_pair(op_device, blob_device);
          }
        }
      };

      // Check that inputs have same device type as the op
      for (int i = 0; i < op.InputSize(); i++) {
        Check(op.InputBlob(i), op_def.input(i));
      }
      for (int i = 0; i < op.OutputSize(); i++) {
        Check(*op.OutputBlob(i), op_def.output(i));
      }
      return mismatches;
    */
}
