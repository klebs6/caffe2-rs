crate::ix!();

pub struct TensorInfo {
    dims:                 Vec<u64>,
    onnxifi_type:         u64,
    quantized:            bool,
    quantization_axis:    u32,
    quantization_params:  u64,
    scales:               Vec<f32>,
    biases:               Vec<i32>,
}

impl From<&TensorProto> for TensorInfo {
    fn from(x: &TensorProto) -> Self {
        todo!();
        /*
            : onnxifi_type(getOnnxifiDataType(t.data_type())),
          quantized(false),
          quantizationAxis(0),
          quantizationParams(0) 

      for (const auto d : t.dims()) {
        dims.push_back(d);
      }
        */
    }
}

impl From<&QTensorProto> for TensorInfo {
    fn from(x: &QTensorProto) -> Self {

        todo!();
        /*
            : onnxifi_type(getOnnxifiDataType(t.data_type())),
          quantized(true),
          quantizationAxis(t.has_axis() ? t.axis() : 0),
          quantizationParams(t.scales_size() ? t.scales_size() : 1) 

      for (const auto d : t.dims()) {
        dims.push_back(d);
      }
      if (t.scales_size()) {
        for (const auto d : t.scales()) {
          scales.push_back(static_cast<float>(d));
        }
        for (const auto d : t.biases()) {
          biases.push_back(static_cast<int32_t>(d));
        }
      } else {
        scales.push_back(static_cast<float>(t.scale()));
        biases.push_back(static_cast<int32_t>(t.bias()));
      }
        */
    }
}
