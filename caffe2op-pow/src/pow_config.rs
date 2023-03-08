crate::ix!();

num_inputs!{Pow, (1,2)}

num_outputs!{Pow, 1}

inputs!{Pow, 
    0 => ("X", "Input data blob to be operated on."),
    1 => ("exponent", "Exponent blob containing the exponent(s) for calculation. Do not use if setting exponent via argument.")
}

outputs!{Pow, 
    0 => ("Y", "Output data blob with the same shape as the input.")
}

args!{Pow, 
    0 => ("exponent", "The exponent of the power function. Do not use if setting exponent via input."),
    1 => ("axis", "*(type: int; default: -1)*"),
    2 => ("broadcast", "*(type: bool; default: False)*")
}

identical_type_and_shape_of_input!{Pow, 0}

allow_inplace!{Pow, vec![(0, 0), (1, 0)]}

register_cpu_operator!{
    Pow,
    PowOp<
    TensorTypes<f32>, 
    NumericTypes,
    CPUContext,
    EigenPowFunctor,
    SameTypeAsInput>
}
