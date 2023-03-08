crate::ix!();

num_inputs!{PackSegments, 2}

num_outputs!{PackSegments, (1,2)}

inputs!{PackSegments, 
    0 => ("lengths", "1-d int/long tensor contains the length in each of the output."),
    1 => ("tensor",  "N dim Tensor.")
}

outputs!{PackSegments, 
    0 => ("packed_tensor", "N + 1 dim Tensor where dim(1) is the max length, dim(0) is the batch size."),
    1 => ("presence_mask", "2 dim boolean tensor, false where packed_tensor is padded, true otherwise.")
}

args!{PackSegments, 
    0 => ("max_length",           "The pre-defined max_length for the packed segments"),
    1 => ("pad_minf",             "Padding number in the packed segments. Use true to pad -infinity, otherwise pad zeros"),
    2 => ("return_presence_mask", "bool whether to return presence mask, false by default")
}

input_tags!{
    PackSegmentsOp {
        Lengths,
        Data
    }
}
