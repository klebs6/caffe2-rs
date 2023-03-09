crate::ix!();

input_tags!{
    SparseLpRegularizerOp {
        Param,
        Indices
    }
}

output_tags!{
    SparseLpRegularizerOp {
        OutputParam
    }
}

register_cpu_operator!{
    SparseLpRegularizer,
    SparseLpRegularizerOp<f32, CPUContext>}

num_inputs!{SparseLpRegularizer, (2,3)}

num_outputs!{SparseLpRegularizer, 1}

inputs!{SparseLpRegularizer, 
    0 => ("param",   "Parameters to be regularized"),
    1 => ("indices", "Sparse indices"),
    2 => ("grad",    "Gradient computed (optional - not used, this argument is for backwards compatibility)")
}

outputs!{SparseLpRegularizer, 
    0 => ("output_param", "Regularized parameters")
}

args!{SparseLpRegularizer, 
    0 => ("p",          "Value of p in the Lp regularization to use. The default is 2.0."),
    1 => ("reg_lambda", "Value of lambda (multiplier for the regularization term). The default is 1e-5.")
}

enforce_one_to_one_inplace!{SparseLpRegularizer}

should_not_do_gradient!{SparseLpNorm}
