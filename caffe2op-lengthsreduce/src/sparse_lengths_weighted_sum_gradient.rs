crate::ix!();

register_cpu_operator!{
    SparseLengthsWeightedSumGradient,
    SparseLengthsWeightedSumDef::BackwardOp
}

num_inputs!{SparseLengthsWeightedSumGradient, SparseLengthsWeightedSumDef::BackwardOp::kNumInputs}

num_outputs!{SparseLengthsWeightedSumGradient, 1}

disallow_input_fillers!{SparseLengthsWeightedSumGradient}

register_gradient!{
    SparseLengthsWeightedSum,
    SparseLengthsWeightedSumDef::GetGradient
}
