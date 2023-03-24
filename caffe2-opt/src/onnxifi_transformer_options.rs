crate::ix!();

pub struct OnnxifiTransformerOptions {

    base: BackendTransformOptions,

    /**
      | Pass serialized onnx model if true,
      | otherwise pass serialized c2 model
      |
      */
    use_onnx:                       bool, // default = false

    /**
      | Whether to adjust batch at the outputs
      | or not
      |
      */
    adjust_batch:                   bool, // default = true

    /// Whether to lower model blob by blob
    load_model_by_blob:             bool, // default = false

    /// Whether to enforce fp32 inputs into fp16.
    enforce_fp32_inputs_into_fp16:  bool, // default = false

    /**
      | Whether to combine fp32 batched inputs
      | into one tensor and convert it to fp16
      | or not
      |
      */
    merge_fp32_inputs_into_fp16:    bool, // default = false

    /// Whether the net has been ssaRewritten
    predictor_net_ssa_rewritten:    bool, // default = false

    /// Inference timeout
    timeout:                        i32, // default = 0

    /// Mapping of batch sizes to shape infos
    shape_hints_per_bs:             HashMap<i32,ShapeInfoMap>,
}

impl OnnxifiTransformerOptions {
    
    pub fn new() -> Self {
    
        todo!();
        /*
            : BackendTransformOptions()
        */
    }
}

