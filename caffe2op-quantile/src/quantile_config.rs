crate::ix!();

register_cpu_operator!{Quantile, QuantileOp<CPUContext>}

num_inputs!{Quantile, (1,INT_MAX)}

num_outputs!{Quantile, 1}

inputs!{Quantile, 
    0 => ("X1, X2, ...", "*(type: Tensor`<float>`)* List of input tensors.")
}

outputs!{Quantile, 
    0 => ("quantile_value", "Value at the given quantile")
}

args!{Quantile, 
    0 => ("abs", "If true (default), apply abs() on the tensor values."),
    1 => ("tol", "multiplicative tolerance of the quantile_value.")
}

should_not_do_gradient!{Quantile}

output_tags!{
    QuantileOp {
        QuantileVal
    }
}
