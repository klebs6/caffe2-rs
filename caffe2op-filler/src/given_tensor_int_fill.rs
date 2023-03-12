crate::ix!();

num_inputs!{GivenTensorIntFill, (0,1)}

num_outputs!{GivenTensorIntFill, 1}

args!{GivenTensorIntFill, 
    0 => ("values",           "The value for the elements of the output tensor. true /* required */"),
    1 => ("shape",            "The shape of the output tensor. Cannot set the shape argument and pass in an input at the same time."),
    2 => ("extra_shape",      "The additional dimensions appended at the end of the shape indicated by the input blob. Cannot set the extra_shape argument when there is no input blob."),
    3 => ("input_as_shape",   "1D tensor containing the desired output shape. First input must be in CPU context.")
}

tensor_inference_function!{GivenTensorIntFill, 
    FillerTensorInference::<TensorProto_DataType_INT32> 
}

allow_inplace!{GivenTensorIntFill, vec![(0, 0)]}
