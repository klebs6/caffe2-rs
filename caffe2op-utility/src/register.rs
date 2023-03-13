crate::ix!();

register_cpu_operator!{WallClockTime, WallClockTimeOp<CPUContext>}
register_cpu_operator!{Print, PrintOp<CPUContext>}
register_cpu_operator!{FlattenToVec, FlattenToVecOp<CPUContext>}
register_cpu_operator!{Alias, AliasOp<CPUContext>}
register_cpu_operator!{ResizeLike, ResizeLikeOp<CPUContext>}
register_cpu_operator!{SumInt, SumOp<CPUContext>}
register_cpu_operator!{WeightedSum, WeightedSumOp<CPUContext>}
register_cpu_operator!{WeightedSumGradient, WeightedSumGradientOp<CPUContext>}
register_cpu_operator!{ScatterWeightedSum, ScatterWeightedSumOp<float, CPUContext>}
register_cpu_operator!{ScatterAssign, ScatterAssignOp<CPUContext>}
register_cpu_operator!{Scatter, ScatterOp<CPUContext>}

register_cpu_operator!{LengthsToShape, LengthsToShapeOp<CPUContext>}
register_cpu_operator!{HasElements, HasElementsOp<CPUContext>}
register_cpu_operator!{GatherRanges, GatherRangesOp<CPUContext>}
register_cpu_operator!{LengthsGather, LengthsGatherOp<CPUContext>}
register_cpu_operator!{LengthsToSegmentIds, LengthsToSegmentIdsOp<CPUContext>}
register_cpu_operator!{LengthsToRanges, LengthsToRangesOp<CPUContext>}
register_cpu_operator!{LengthsToOffsets, LengthsToOffsetsOp<CPUContext>}
register_cpu_operator!{SegmentIdsToLengths, SegmentIdsToLengthsOp<CPUContext>}
register_cpu_operator!{SegmentIdsToRanges, SegmentIdsToRangesOp<CPUContext>}
register_cpu_operator!{LengthsToWeights, LengthsToWeightsOp<CPUContext>}
register_cpu_operator!{EnsureDense, EnsureDenseOp<CPUContext>}
register_cpu_operator!{AccumulateHistogram, AccumulateHistogramOp<float, CPUContext>}

register_gradient!{WeightedSum, GetWeightedSumGradient}
register_gradient!{FlattenToVec, GetFlattenToVecGradient}

should_not_do_gradient!{LengthsToSegmentIds}
should_not_do_gradient!{SegmentIdsToLengths}
should_not_do_gradient!{SegmentIdsToRanges}
should_not_do_gradient!{SegmentIdsToLengthWeights}
should_not_do_gradient!{GatherRangesOp}
should_not_do_gradient!{LengthsGather}
should_not_do_gradient!{AccumulateHistogram}
