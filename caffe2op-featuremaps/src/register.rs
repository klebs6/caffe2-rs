crate::ix!();

register_cpu_operator!{MergeMultiListFeatureTensorsGradient,       MergeMultiListOrMapFeatureTensorsGradientOp<CPUContext>}
register_cpu_operator!{MergeMultiMapFeatureTensors,                MergeMultiMapFeatureTensorsOp<CPUContext>}
register_cpu_operator!{MergeMultiMapFeatureTensorsGradient,        MergeMultiListOrMapFeatureTensorsGradientOp<CPUContext>}
register_cpu_operator!{MergeMultiScalarFeatureTensors,             MergeMultiScalarFeatureTensorsOp<CPUContext>}
register_cpu_operator!{MergeMultiScalarFeatureTensorsGradient,     MergeMultiScalarFeatureTensorsGradientOp<CPUContext>}
register_cpu_operator!{MergeSingleListFeatureTensors,              MergeSingleListFeatureTensorsOp<CPUContext>}
register_cpu_operator!{MergeSingleListFeatureTensorsGradient,      MergeSingleListOrMapFeatureTensorsGradientOp<CPUContext>}
register_cpu_operator!{MergeSingleMapFeatureTensors,               MergeSingleMapFeatureTensorsOp<CPUContext>}
register_cpu_operator!{MergeSingleMapFeatureTensorsGradient,       MergeSingleListOrMapFeatureTensorsGradientOp<CPUContext>}
register_cpu_operator!{MergeSingleScalarFeatureTensorsGradient,    MergeSingleScalarFeatureTensorsGradientOp<CPUContext>}
register_cpu_operator!{MergeDenseFeatureTensors,                   MergeDenseFeatureTensorsOp<CPUContext>}
register_cpu_operator!{MergeSingleScalarFeatureTensors,            MergeSingleScalarFeatureTensorsOp<CPUContext>}

register_gradient!{MergeMultiListFeatureTensors,                   GetMergeMultiListFeatureTensorsGradient}
register_gradient!{MergeMultiMapFeatureTensors,                    GetMergeMultiMapFeatureTensorsGradient}
register_gradient!{MergeMultiScalarFeatureTensors,                 GetMergeMultiScalarFeatureTensorsGradient}
register_gradient!{MergeSingleListFeatureTensors,                  GetMergeSingleListFeatureTensorsGradient}
register_gradient!{MergeSingleMapFeatureTensors,                   GetMergeSingleMapFeatureTensorsGradient}
register_gradient!{MergeSingleScalarFeatureTensors,                GetMergeSingleScalarFeatureTensorsGradient}
