crate::ix!();

num_inputs!{SparseToDenseMaskGradient, (2,3)}

num_outputs!{SparseToDenseMaskGradient, 1}

// TODO: enable the filler
disallow_input_fillers!{SparseToDenseMaskGradient}

input_tags!{
    SparseToDenseMaskGradientOp {
        Indices,
        Goutput,
        Lengths
    }
}

output_tags!{
    SparseToDenseMaskGradientOp {
        Gvalues
    }
}

register_cpu_operator!{SparseToDenseMask,         SparseToDenseMaskOp<CPUContext>}

register_cpu_operator!{SparseToDenseMaskGradient, SparseToDenseMaskGradientOp<CPUContext>}
