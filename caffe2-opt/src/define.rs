crate::ix!();

declare_string!{onnxifi_blacklist}

declare_string!{onnxifi_blacklist_ops}

define_bool!{onnxifi_debug_mode, false, "Enable onnxifi debug mode."}

define_bool!{onnxifi_adjust_batch,
    true,
    "Attach AdjustBatch ops at input/outputs of the Onnxifi ops"}

define_bool!{enforce_fp32_inputs_into_fp16,
    false,
    "Whether to enforce fp32 to fp16 conversion for external inputs."}

define_bool!{merge_fp32_inputs_into_fp16,
    false,
    "Merge all the fp32 input tensors into one, convert it to fp16 and split it back"}

define_int32!{
    onnxifi_min_ops,
    1,
    "Minimum number of ops for a subgraph to be lowered to backend"
}

define_int32!{
    onnxifi_timeout_ms,
    0,
    "Timeout limit for onnxifi inference in milliseconds. 0 means no timeout"
}

define_string!{
    onnxifi_shape_hints,
    "",
    "Shape hints in the form of Name:d0,d1:d2;"
}

define_string!{
    onnxifi_blacklist,
    "",
    "A list of net positions whose corresponding op will be 
        ignored to onnxifi. Example 0-50,61,62-70"
}

define_string!{
    onnxifi_blacklist_ops,
    "",
    "A list of operator types that will be ignored to onnxifi. Example Tanh,Mul"
}

define_string!{
    onnxifi_input_output_observe_list,
    "",
    "A list of net positions whose corresponding op's inputs and outputs will be observed. "
}
