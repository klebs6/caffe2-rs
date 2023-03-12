crate::ix!();

num_inputs!{GivenTensorStringFill, (0,1)}

num_outputs!{GivenTensorStringFill, 1}

args!{GivenTensorStringFill, 
    0 => ("values",           "The value for the elements of the output tensor. true /* required */"),
    1 => ("shape",            "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    2 => ("extra_shape",      "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    3 => ("input_as_shape",   "1D tensor containing the desired output shape. First input must be in CPU context.")
}

tensor_inference_function!{
    GivenTensorStringFill, 
    FillerTensorInference::<TensorProto_DataType_STRING>
}

allow_inplace!{GivenTensorStringFill, vec![(0, 0)]}
