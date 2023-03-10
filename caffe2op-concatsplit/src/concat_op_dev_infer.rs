crate::ix!();

#[inline] pub fn concat_op_dev_infer(def: &OperatorDef) -> (Vec<DeviceOption>,Vec<DeviceOption>) {
    
    todo!();
    /*
        auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);

      // 2nd output's type is always CPU irrespective of op's device option.
      CAFFE_ENFORCE_GT(out_dev.size(), 1);
      out_dev[1] = DeviceOption();
      return std::make_pair(in_dev, out_dev);
    */
}
