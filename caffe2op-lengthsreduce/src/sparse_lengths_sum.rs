crate::ix!();

/**
  | Use _STR option because the schema is
  | declared using _STR version too in generic
  | fashion. Otherwise it'd break schema
  | declaration check.
  | 
  | TODO(dzhulgakov): remove _STR when
  | all lengths ops are off generic version.
  |
  */
pub type SparseLengthsSumOp         = CPUSparseLengthsReductionOp<f32, TensorTypes<(f32, f16)>, false, false>;
pub type SparseLengthsWeightedSumOp = CPUSparseLengthsReductionOp<f32, TensorTypes<(f32, f16)>, true, false>;
pub type SparseLengthsMeanOp        = CPUSparseLengthsReductionOp<f32, TensorTypes<(f32, f16)>, false, true>;

register_cpu_operator!{SparseLengthsSum,         SparseLengthsSumOp}
register_cpu_operator!{SparseLengthsWeightedSum, SparseLengthsWeightedSumOp}
register_cpu_operator!{SparseLengthsMean,        SparseLengthsMeanOp}

/**
  | Variation of SparseLengthsWeightedSum
  | operator, where, for each row, weights
  | are accessed by indices [0..L-1], where
  | L is the length of given row.
  | 
  | This is basically a fused operator of
  | LengthsRangeFill + Gather +
  | 
  | SparseWeightedSum
  |
  */
register_cpu_operator_str!{
    "SparseLengthsPositionalWeightedSum",
    CPUSparseLengthsReductionOp::<f32, TensorTypes::<f32, f16>, 1, 0, 1>
}

num_inputs!{SparseLengthsPositionalWeightedSum, 4}

num_outputs!{SparseLengthsPositionalWeightedSum, 1}

inputs!{SparseLengthsPositionalWeightedSum, 
    0 => ("DATA",      "uint8 tensor obtained with operator FloatToRowwiseQuantized8Bits"),
    1 => ("WEIGHT",    "Scalar multipliers for the input slices. Must be a vector with the length matching the length of DATA"),
    2 => ("INDICES",   "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS",   "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsPositionalWeightedSum, 
    0 => ("output",    "output")
}

pub const GradientNeedIndices: bool = true;

pub type SparseLengthsSumDef = AbstractSparseLengthsDef<
    f32,
    i32,
    CPUContext,
    SumReducerDef,
    GradientNeedIndices>;

num_inputs!{SparseLengthsSum, SparseLengthsSumDef::ForwardOp::kNumInputs}

num_outputs!{SparseLengthsSum, 1}

outputs!{SparseLengthsSum, 
    0 => ("OUTPUT", "Aggregated tensor")
}

inherit_onnx_schema!{SparseLengthsSum}

value_key_length_input_fillers!{
    /*
    SparseLengthsSum, 
    ( SparseLengthsSumOp::DATA, SparseLengthsSumOp::INDICES, SparseLengthsSumOp::LENGTHS)
        */
}

fill_using!{
    /*
    SparseLengthsSum, SparseLengthsSumDef::PopulateSchema
        */
}

///-----------------------
register_cpu_operator!{SparseLengthsSumGradient, SparseLengthsSumDef::BackwardOp}

num_inputs!{SparseLengthsSumGradient, SparseLengthsSumDef::BackwardOp::kNumInputs}

num_outputs!{SparseLengthsSumGradient, 1}

disallow_input_fillers!{SparseLengthsSumGradient}

register_gradient!{SparseLengthsSum, SparseLengthsSumDef::GetGradient}

