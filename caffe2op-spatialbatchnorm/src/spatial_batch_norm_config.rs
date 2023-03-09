crate::ix!();

register_cpu_operator!{SpatialBN, SpatialBNOp<CPUContext>}

num_inputs!{SpatialBN, (5,7)}

num_outputs!{SpatialBN, (1,5)}

inputs!{SpatialBN, 
    0 => ("X",      "The input 4-dimensional tensor of shape $NCHW$ or $NHWC$ depending on the order parameter."),
    1 => ("scale",  "The scale as a 1-dimensional tensor of size $C$ to be applied to the output."),
    2 => ("bias",   "The bias as a 1-dimensional tensor of size $C$ to be applied to the output."),
    3 => ("mean",   "The running mean (training) or the estimated mean (testing) as a 1-dimensional tensor of size $C$."),
    4 => ("var",    "The running variance (training) or the estimated variance (testing) as a 1-dimensional tensor of size $C$."),
    5 => ("sums",   "*(optional)* Per-channel sums of elements to be used to determine the mean and variance for this batch."),
    6 => ("sumsq",  "*(optional)* Per-channel sum of elements squared per channel to be used to determine the variance for this batch.")
}

outputs!{SpatialBN, 
    0 => ("Y",            "The output 4-dimensional tensor of the same shape as $X$."),
    1 => ("mean",         "The running mean after the spatial BN operator. Must be in-place with the input *mean*. Should not be used for testing."),
    2 => ("var",          "The running variance after the spatial BN operator. Must be in-place with the input *var*. Should not be used for testing."),
    3 => ("saved_mean",   "Saved mean used during training to speed up gradient computation. Should not be used for testing."),
    4 => ("saved_var",    "Saved variance used during training to speed up gradient computation. Should not be used for testing.")
}

args!{SpatialBN, 

    0 => ("epsilon",      
        "*(type: float; default: 1e-5)* The epsilon value to use to avoid division by zero."),

    1 => ("order",        
        "*(type: string; default: NCHW)* Specifies the order of the input data blob, where $N$ is batch size, 
        $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is NHWC."),

    2 => ("momentum",     
        "*(type: float; default: 0.9)* Factor used in computing the running mean and variance. 
        e.g., running_mean = running_mean x momentum + mean x (1 - momentum)"),

    3 => ("num_batches",  
        "*(type: int; default: 1)* Specifies the number of batches to apply normalization on. 
        Requires specifying the optional sums and sumsq inputs that provide statistics across multiple 
        batches from which mean and variance can be determined.")
}

arg_is_test!{SpatialBN, "*(type: int; default: 0)* If set to nonzero, run spatial batch normalization in test mode."}

inherit_onnx_schema!{SpatialBN, "BatchNormalization"}

allow_inplace!{SpatialBN,   vec![(0, 0), (5, 3), (6, 4)]}

enforce_inplace!{SpatialBN, vec![(3, 1), (4, 2)]}

cost_inference_function!{SpatialBN, CostInferenceForSpatialBN}

input_tags!{
    SpatialBNOp {
        Input,
        Scale,
        Bias,
        EstMean,
        EstVar,
        BatchMeanSum,
        BatchVarSum
    }
}

output_tags!{
    SpatialBNOp {
        Output,
        RunningMean,
        RunningVar,
        SavedMean,
        SavedInvStd
    }
}
