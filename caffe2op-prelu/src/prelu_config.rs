crate::ix!();

num_inputs!{PRelu, 2}

num_outputs!{PRelu, 1}

inputs!{PRelu, 
    0 => ("X", "Input tensor of data to be operated on."),
    1 => ("Slope", "1D input slope tensor. If `Slope` is of size 1, the value is shared across different channels")
}

outputs!{PRelu, 
    0 => ("Y", "Output tensor, with same shape as $X$.")
}

identical_type_and_shape_of_input!{PRelu, 0}

allow_inplace!{PRelu, vec![(0, 0)]}

inherit_onnx_schema!{PRelu}
