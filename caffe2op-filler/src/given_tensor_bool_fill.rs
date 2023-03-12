crate::ix!();

num_inputs!{GivenTensorBoolFill, (0,1)}

num_outputs!{GivenTensorBoolFill, 1}

args!{GivenTensorBoolFill, 
    0 => ("values",           "The value for the elements of the output tensor. true /* required */"),
    1 => ("shape",            "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    2 => ("extra_shape",      "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    3 => ("input_as_shape",   "1D tensor containing the desired output shape. First input must be in CPU context.")
}

tensor_inference_function!{GivenTensorBoolFill, 
    FillerTensorInference::<TensorProto_DataType_BOOL> 
}

allow_inplace!{GivenTensorBoolFill, vec![(0, 0)]}
