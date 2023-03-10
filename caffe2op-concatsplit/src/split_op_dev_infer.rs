crate::ix!();

#[inline] pub fn split_op_dev_infer(def: &OperatorDef) -> (Vec<DeviceOption>, Vec<DeviceOption>) {
    
    todo!();
    /*
        auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);

      // If we obtain split from input tensor, then 2nd input's type is always CPU.
      if (def.input_size() == SplitOp<CPUContext>::kSplitOpInputSize) {
        CAFFE_ENFORCE_GT(in_dev.size(), 1);
        in_dev[1] = DeviceOption();
      }
      return std::make_pair(in_dev, out_dev);
    */
}
