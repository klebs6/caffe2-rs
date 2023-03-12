crate::ix!();

register_cpu_operator!{
    LabelCrossEntropy,
    LabelCrossEntropyOp<f32, CPUContext>
}

register_cpu_operator!{
    LabelCrossEntropyGradient,
    LabelCrossEntropyGradientOp<f32, CPUContext>
}

register_cpu_operator!{SigmoidCrossEntropyWithLogits,                    SigmoidCrossEntropyWithLogitsOp<float, CPUContext>}
register_cpu_operator!{SigmoidCrossEntropyWithLogitsGradient,            SigmoidCrossEntropyWithLogitsGradientOp<float, CPUContext>}
register_cpu_operator!{WeightedSigmoidCrossEntropyWithLogits,            WeightedSigmoidCrossEntropyWithLogitsOp<float, CPUContext>}
register_cpu_operator!{WeightedSigmoidCrossEntropyWithLogitsGradient,    WeightedSigmoidCrossEntropyWithLogitsGradientOp<float, CPUContext>}
register_gradient!{MakeTwoClass,                                         GetMakeTwoClassGradient}
register_gradient!{SigmoidCrossEntropyWithLogits,                        GetSigmoidCrossEntropyWithLogitsGradient}
register_gradient!{WeightedSigmoidCrossEntropyWithLogits,                GetWeightedSigmoidCrossEntropyWithLogitsGradient}
register_cpu_operator!{CrossEntropy,                                     CrossEntropyOp<float, CPUContext>}
register_cpu_operator!{CrossEntropyGradient,                             CrossEntropyGradientOp<float, CPUContext>}
