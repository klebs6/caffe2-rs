crate::ix!();

register_cpu_operator!{UniformFill,         UniformFillOp<f32, CPUContext>}
register_cpu_operator!{UniformIntFill,      UniformFillOp<i32, CPUContext>}
register_cpu_operator!{UniqueUniformFill,   UniqueUniformFillOp<CPUContext>}
register_cpu_operator!{ConstantFill,        ConstantFillOp<CPUContext>}
register_cpu_operator!{DiagonalFill,        DiagonalFillOp<CPUContext>}
register_cpu_operator!{GaussianFill,        GaussianFillOp<f32, CPUContext>}
register_cpu_operator!{XavierFill,          XavierFillOp<f32, CPUContext>}
register_cpu_operator!{MSRAFill,            MSRAFillOp<f32, CPUContext>}
register_cpu_operator!{RangeFill,           RangeFillOp<f32, CPUContext>}
register_cpu_operator!{LengthsRangeFill,    LengthsRangeFillOp<CPUContext>}

no_gradient!{UniformFill}
no_gradient!{UniformIntFill}
no_gradient!{UniqueUniformFill}
no_gradient!{ConstantFill}
no_gradient!{DiagonalFill}
no_gradient!{GaussianFill}
no_gradient!{XavierFill}
no_gradient!{MSRAFill}
no_gradient!{RangeFill}
no_gradient!{LengthsRangeFill}

register_cpu_operator!{GivenTensorFill,         GivenTensorFillOp<f32,    CPUContext>}
register_cpu_operator!{GivenTensorDoubleFill,   GivenTensorFillOp<f64,    CPUContext>}
register_cpu_operator!{GivenTensorBoolFill,     GivenTensorFillOp<bool,   CPUContext>}
register_cpu_operator!{GivenTensorInt16Fill,    GivenTensorFillOp<i16,    CPUContext>}
register_cpu_operator!{GivenTensorIntFill,      GivenTensorFillOp<i32,    CPUContext>}
register_cpu_operator!{GivenTensorInt64Fill,    GivenTensorFillOp<i64,    CPUContext>}
register_cpu_operator!{GivenTensorStringFill,   GivenTensorFillOp<String, CPUContext>}

no_gradient!{GivenTensorFill}
no_gradient!{GivenTensorDoubleFill}
no_gradient!{GivenTensorBoolFill}
no_gradient!{GivenTensorInt16Fill}
no_gradient!{GivenTensorIntFill}
no_gradient!{GivenTensorInt64Fill}
no_gradient!{GivenTensorStringFill}
