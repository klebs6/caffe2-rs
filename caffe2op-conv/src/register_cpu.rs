crate::ix!();

register_cpu_operator_with_engine!{Conv,   EIGEN, EigenConvOp::<f32>}
register_cpu_operator_with_engine!{Conv1D, EIGEN, EigenConvOp::<f32>}
register_cpu_operator_with_engine!{Conv2D, EIGEN, EigenConvOp::<f32>}
register_cpu_operator_with_engine!{Conv3D, EIGEN, EigenConvOp::<f32>}
