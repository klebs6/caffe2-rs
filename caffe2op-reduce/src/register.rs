crate::ix!();

register_cpu_operator!{SumElements,           SumElementsOp<float, CPUContext>}
register_cpu_operator!{SumElementsInt,        SumElementsIntOp<int, CPUContext>}
register_cpu_operator!{SumSqrElements,        SumSqrElementsOp<CPUContext>}
register_cpu_operator!{SumElementsGradient,   SumElementsGradientOp<float, CPUContext>}
register_cpu_operator!{RowwiseMaxGradient,    MaxReductionGradientOp<float, CPUContext, true>}
register_cpu_operator!{ColwiseMaxGradient,    MaxReductionGradientOp<float, CPUContext, false>}
