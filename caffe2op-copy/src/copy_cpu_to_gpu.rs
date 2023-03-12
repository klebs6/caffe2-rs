crate::ix!();

/**
  | Copy tensor for CPU to GPU context. Must
  | be run under GPU device option.
  |
  */
pub type CopyCPUToGPUOp = CopyOp<CPUContext, CPUContext, GPUContext>;

num_inputs!{CopyCPUToGPU, 1}

num_outputs!{CopyCPUToGPU, 1}

inputs!{CopyCPUToGPU, 
    0 => ("input", "The input tensor.")
}

outputs!{CopyCPUToGPU, 
    0 => ("output", "Tensor that will contain a copy of the input.")
}

identical_type_and_shape!{CopyCPUToGPU}

inputs_can_cross_devices!{CopyCPUToGPU}

device_inference_function!{CopyCPUToGPU, 
    |def: &OperatorDef| {
        todo!();
        /*
          CAFFE_ENFORCE( def.has_device_option(), "CopyCPUToGPU op should have cuda device option.");
          auto& cuda_option = def.device_option();
          auto cpu_option = DeviceOption();
          vector<DeviceOption> in_dev(def.input_size(), cpu_option);
          vector<DeviceOption> out_dev(def.output_size(), cuda_option);
          return std::make_pair(in_dev, out_dev);
        */
    }
}
