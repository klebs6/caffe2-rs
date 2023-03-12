crate::ix!();

register_cpu_operator!{SparseLengthsMeanGradient, SparseLengthsMeanDef::BackwardOp}

num_inputs!{SparseLengthsMeanGradient, SparseLengthsMeanDef::BackwardOp::kNumInputs}

num_outputs!{SparseLengthsMeanGradient, 1}

disallow_input_fillers!{SparseLengthsMeanGradient}

register_gradient!{SparseLengthsMean, SparseLengthsMeanDef::GetGradient}

num_inputs!{TTSparseLengthsSumGradient, 8}

num_outputs!{TTSparseLengthsSumGradient, 3}
