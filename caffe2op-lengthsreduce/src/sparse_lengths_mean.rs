crate::ix!();

pub type SparseLengthsMeanDef = AbstractSparseLengthsDef<f32, i32, CPUContext, MeanReducerDef, GradientNeedIndices>;

num_inputs!{SparseLengthsMean, SparseLengthsMeanDef::ForwardOp::kNumInputs}

num_outputs!{SparseLengthsMean, 1}

outputs!{SparseLengthsMean, 
    0 => ("OUTPUT", "Aggregated tensor")
}

value_key_length_input_fillers!{
    /*
    SparseLengthsMean, (
        SparseLengthsMeanOp::DATA,
        SparseLengthsMeanOp::INDICES,
        SparseLengthsMeanOp::LENGTHS
    )
    */
}

fill_using!{
    /*
    SparseLengthsMean, SparseLengthsMeanDef::PopulateSchema
        */
}


