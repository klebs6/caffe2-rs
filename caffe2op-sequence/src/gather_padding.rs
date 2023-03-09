crate::ix!();

/**
  | Gather the sum of start and end paddings
  | in a padded input sequence. Used in order
  | to compute the gradients of AddPadding
  | w.r.t the padding tensors.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GatherPaddingOp<Context> {

    storage:                    OperatorStorage,
    context:                    Context,

    start_padding_width:        i32,
    end_padding_width:          i32,

    // Scratch space required by the CUDA version
    lengths_prefix_sum_buffer:  Tensor, // {Context::GetDeviceType()};
    lengths_prefix_sum:         Tensor, // {Context::GetDeviceType()};
}

num_inputs!{GatherPadding, 2}

num_outputs!{GatherPadding, (1,2)}

inputs!{GatherPadding, 
    0 => ("data_in", "T<N, D1..., Dn> Padded input data"),
    1 => ("lengths", "(i64) Num of elements in each range. sum(lengths) = N. If not provided, considers all data as a single segment.")
}

outputs!{GatherPadding, 
    0 => ("padding_sum", "Sum of all start paddings, or of all paddings if end_padding_sum is not provided."),
    1 => ("end_padding_sum", "T<D1..., Dn> Sum of all end paddings, if provided.")
}

args!{GatherPadding, 
    0 => ("padding_width", "Outer-size of padding present around each range."),
    1 => ("end_padding_width", "(Optional) Specifies a different end-padding width.")
}
