crate::ix!();

/**
  | Element-wise modulo operation. Each
  | element in the output is the modulo result
  | of the corresponding element in the
  | input data. The divisor of the modulo
  | is provided by the `divisor` argument.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mod_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ModOp<Context> {
    storage:             OperatorStorage,
    context:             Context,
    divisor:             i64,
    sign_follow_divisor: bool,
}

register_cpu_operator!{Mod, ModOp<CPUContext>}

num_inputs!{Mod, 1}

num_outputs!{Mod, 1}

inputs!{Mod, 
    0 => ("X", "*(type: Tensor`<int>`)* Input tensor with int32 or int64 data.")
}

outputs!{Mod, 
    0 => ("Y", "*(type: Tensor`<int>`)* Output tensor of data with modulo operation applied.")
}

args!{Mod, 
    0 => ("divisor", "*(type: int; default: 0)* Divisor of the modulo operation (must be >= 1)."),
    1 => ("sign_follow_divisor", "*(type: bool; default: False)* If true, sign of output matches divisor, else if false, sign follows dividend.")
}

identical_type_and_shape!{Mod}

allow_inplace!{Mod, vec![(0, 0)]}

should_not_do_gradient!{ModOp}

input_tags!{
    ModOp {
        Data
    }
}
