crate::ix!();

register_cpu_operator!{
    SpatialBNGradient, 
    SpatialBNGradientOp<CPUContext>
}

register_gradient!{
    SpatialBN, 
    GetSpatialBNGradient
}

num_inputs!{SpatialBNGradient, (5,7)}

num_outputs!{SpatialBNGradient, 3}

allow_inplace!{SpatialBNGradient, vec![(5, 1), (6, 2)]}

input_tags!{
    SpatialBNGradientOp {
        Input,
        Scale,
        OutputGrad,
        SavedMean,
        SavedInvStd,
        AggregateScaleGrad,
        AggregateBiasGrad
    }
}

output_tags!{
    SpatialBNGradientOp {
        InputGrad,
        ScaleGrad,
        BiasGrad
    }
}
