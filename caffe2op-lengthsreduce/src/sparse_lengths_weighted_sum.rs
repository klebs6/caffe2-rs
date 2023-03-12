crate::ix!();

pub type SparseLengthsWeightedSumDef = AbstractSparseLengthsDef<f32, i32, CPUContext, WeightedSumReducerDef, GradientNeedIndices>;

num_inputs!{SparseLengthsWeightedSum, SparseLengthsWeightedSumDef::ForwardOp::kNumInputs}

num_outputs!{SparseLengthsWeightedSum, 1}

outputs!{SparseLengthsWeightedSum, 
    0 => ("OUTPUT", "Aggregated tensor")
}

inherit_onnx_schema!{SparseLengthsWeightedSum}

weighted_value_key_length_input_fillers!{
    /*
    SparseLengthsWeightedSum, ( SparseLengthsWeightedSumOp::DATA, SparseLengthsWeightedSumOp::INDICES, SparseLengthsWeightedSumOp::LENGTHS, SparseLengthsWeightedSumOp::WEIGHT)
        */
}

fill_using!{
    /*
    SparseLengthsWeightedSum, SparseLengthsWeightedSumDef::PopulateSchema
        */
}
