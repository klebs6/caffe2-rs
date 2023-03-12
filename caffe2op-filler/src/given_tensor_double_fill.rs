crate::ix!();

num_inputs!{GivenTensorDoubleFill, (0,1)}

num_outputs!{GivenTensorDoubleFill, 1}

args!{GivenTensorDoubleFill, 
    0 => ("values",            "The value for the elements of the output tensor. true /* required */"),
    1 => ("shape",             "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    2 => ("extra_shape",       "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    3 => ("input_as_shape",    "1D tensor containing the desired output shape. First input must be in CPU context.")
}

tensor_inference_function!{GivenTensorDoubleFill, 
    FillerTensorInference::<TensorProto_DataType_DOUBLE> 
}

allow_inplace!{GivenTensorDoubleFill, vec![(0, 0)]}

