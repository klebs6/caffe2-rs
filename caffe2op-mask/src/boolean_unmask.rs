crate::ix!();

/**
  | Given a series of masks and values,
  | reconstruct values together according
  | to masks.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_unmask_ops.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BooleanUnmaskOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{
    BooleanUnmask, 
    BooleanUnmaskOp<CPUContext>
}

num_outputs!{BooleanUnmask, 1}

inputs!{BooleanUnmask, 
    0 => ("data", "(*Tensor*): 1D input tensor(s)"),
    1 => ("mask", "(*Tensor`<bool>`*): 1D boolean mask tensor(s)")
}

outputs!{BooleanUnmask, 
    0 => ("unmasked_data", "(*Tensor*): 1D tensor of same type as `data` input that contains the unmasked input tensor")
}

num_inputs!{
    BooleanUnmask, |n: i32| -> bool { (n > 0) && (n % 2 == 0) }
}

no_gradient!{BooleanUnmask}
