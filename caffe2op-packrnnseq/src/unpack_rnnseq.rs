crate::ix!();

/**
 | This is the reverse operator for
 | PackRNNSequence. It maps the packed values back to
 | sequence values based on the length blob. Each
 | number from length blob represents the
 | corresponding values that has been grouped. The
 | dimension for each pack is the same as the maximum
 | number from the length blob (padding with zero was
 | implemented for smaller length value). The overall
 | output dimension is: M * D, where M is the sum of
 | lengths, and D is the dimension of each feature
 | value. The following example shows the input and
 | output of this operator:
 |
 |
 | Given:
 |   values = [
 |     [v1, v3, v6, v7],
 |     [v2, v4, 0,  v8],
 |     [0,  v5, 0,  0 ],
 |   ]
 |   lengths = [2, 3, 1, 2]
 |
 |
 | Output:
 |   output = [v1, v2, v3, v4, v5, v6, v7, v8];
 |
 |
 | One application for this operator is the transfer
 | data from the format of RNN back to sequence
 | values. Note that the gradient operator of
 | UnpackRNNSequence is PackRNNSequence.
 */
register_cpu_operator!{
    UnpackRNNSequence,
    PackRNNSequenceOpBase<CPUContext, false>
}

num_inputs!{UnpackRNNSequence, 2}

num_outputs!{UnpackRNNSequence, 1}

inputs!{
    UnpackRNNSequence, 
    0 => ("values", "Data tensor, contains the packed features"),
    1 => ("lengths", "lengths with each number representing the pack size.")
}

outputs!{
    UnpackRNNSequence, 
    0 => ("output", "Output tensor before packing")
}
