crate::ix!();

#[USE_RECURRENT_BASE_FUNCTIONS]
pub struct RecurrentParamAccessOp<T,const mode: RecurrentParamOpMode> {
    base: RecurrentBaseOp<T>,
    phantom: PhantomData<T>,
}

/// Set individual parameters of a recurrent net.
///
register_cudnn_operator!{
    RecurrentParamSet,
    RecurrentParamAccessOp<float, SET_PARAM>
}

num_inputs!{RecurrentParamSet, 3}

num_outputs!{RecurrentParamSet, 1}

inputs!{RecurrentParamSet, 
    0 => ("input",            "Input blob. Needed for inferring the shapes.  A dummy tensor matching the input shape is ok."),
    1 => ("all_params",       "Blob holding all the parameters"),
    2 => ("param",            "Values for the specified parameter")
}

outputs!{RecurrentParamSet, 
    0 => ("all_params",       "Blob holding all the parameters (same as input(1))")
}

args!{RecurrentParamSet, 
    0 => ("param_type",       "Type of param to be set: input_gate_w, forget_gate_w, cell_w, output_gate_w input_gate_b, forget_gate_b, cell_b, output_gate_b"),
    1 => ("input_type",       "'recurrent' or 'input'"),
    2 => ("layer",            "layer index (starting from 0)")
}

enforce_inplace!{RecurrentParamSet, vec![(1, 0)]}

///Retrieve individual parameters of a recurrent net op.
register_cudnn_operator!{
    RecurrentParamGet,
    RecurrentParamAccessOp<float, GET_PARAM>
}

num_inputs!{RecurrentParamGet, 2}

num_outputs!{RecurrentParamGet, 1}

inputs!{RecurrentParamGet, 
    0 => ("input",      "Input blob. Needed for inferring the shapes.  A dummy tensor matching the input shape is ok."),
    1 => ("all_params", "Blob holding all the parameters")
}

outputs!{RecurrentParamGet, 
    0 => ("param", "Blob holding the requested values")
}

args!{RecurrentParamGet, 
    0 => ("param_type", "Type of param to be set: input_gate_w, forget_gate_w, cell_w, output_gate_w input_gate_b, forget_gate_b, cell_b, output_gate_b"),
    1 => ("input_type", "'recurrent' or 'input'"),
    2 => ("layer",      "layer index - starting from 0")
}

impl<T,const mode: RecurrentParamOpMode> RecurrentParamAccessOp<T,mode> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : RecurrentBaseOp<T>(std::forward<Args>(args)...)
        */
    }
}
