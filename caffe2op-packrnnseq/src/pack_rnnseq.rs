crate::ix!();

/**
 | Pack values based on the length blob. Each number
 | from length blob represents the corresponding
 | values that need to be packed. The dimension for
 | each pack is the same as the maximum number from
 | the length blob (padding with zero is implemented
 | for smaller length value). The overall output
 | dimension is:
 |
 | T * N * D, where T is the max number of lengths,
 | N is the size of lengths, and D is the dimension
 | of each feature value. The following example shows
 | the input and output of this operator:
 |
 |
 | Given:
 |   values = [v1, v2, v3, v4, v5, v6, v7, v8]
 |   lengths = [2, 3, 1, 2];
 |
 |
 | Output:
 |   output = [
 |     [v1, v3, v6, v7],
 |     [v2, v4, 0,  v8],
 |     [0,  v5, 0,  0 ],
 |   ]
 |
 |
 | One application for this operator is the transfer
 | data into the format that is used for RNN
 | models. Note that the gradient operator of
 | PackRNNSequence is UnpackRNNSequence.
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PackRNNSequenceOpBase<Context,const Forward: bool> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{PackRNNSequence, 2}

num_outputs!{PackRNNSequence, 1}

inputs!{PackRNNSequence, 
    0 => ("values", "Data tensor, contains a sequence of features"),
    1 => ("lengths", "lengths with each number representing the pack size.")
}

outputs!{PackRNNSequence, 
    0 => ("output", "Output tensor after packing")
}

input_tags!{
    PackRNNSequenceOp {
        Inputvalue,
        Lengths
    }
}

output_tags!{
    PackRNNSequenceOp {
        Outputvalue
    }
}

register_cpu_operator!{
    PackRNNSequence, 
    PackRNNSequenceOpBase<CPUContext, true>
}
