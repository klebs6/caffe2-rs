crate::ix!();

register_cpu_operator!{PairWiseLossGradient, 
    PairWiseLossGradientOp<float, CPUContext>}

num_inputs!{PairWiseLossGradient, (3,4)}

num_outputs!{PairWiseLossGradient, 1}

input_tags!{
    PairWiseLossGradientOp {
        Xvalue,
        Label,
        Dyvalue,
        Lengths
    }
}

output_tags!{
    PairWiseLossGradientOp {
        Dxvalue
    }
}
