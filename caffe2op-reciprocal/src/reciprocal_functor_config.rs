crate::ix!();

num_inputs!{Reciprocal, 1}

num_outputs!{Reciprocal, 1}

inputs!{Reciprocal, 
    0 => ("X", "*(type: Tensor`<f32>`)* Input data tensor.")
}

outputs!{Reciprocal, 
    0 => ("Y", "*(type: Tensor`<f32>`)* Output tensor.")
}

identical_type_and_shape!{Reciprocal}

allow_inplace!{Reciprocal, vec![(0, 0)]}
