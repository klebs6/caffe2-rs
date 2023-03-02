crate::ix!();

///---------------------------------------

caffe_known_type!{Box<TreeCursor>}
caffe_known_type!{TensorVectorPtr}
caffe_known_type!{SharedTensorVectorPtr}
caffe_known_type!{Shared2DTensorVectorPtr}

pub const kDatasetFieldSeparator: &'static str = ":";
pub const kDatasetLengthField:    &'static str = "lengths";

/// how much percent to grow the dataset when needed
pub const kDatasetGrowthPct: i32 = 40;

register_cpu_operator!{CreateTreeCursor,          CreateTreeCursorOp}
register_cpu_operator!{ResetCursor,               ResetCursorOp}
register_cpu_operator!{ReadNextBatch,             ReadNextBatchOp}
register_cpu_operator!{GetCursorOffset,           GetCursorOffsetOp}
register_cpu_operator!{ComputeOffset,             ComputeOffsetOp}
register_cpu_operator!{SortAndShuffle,            SortAndShuffleOp}
register_cpu_operator!{ReadRandomBatch,           ReadRandomBatchOp}
register_cpu_operator!{CheckDatasetConsistency,   CheckDatasetConsistencyOp}
register_cpu_operator!{Append,                    AppendOp<CPUContext>}
register_cpu_operator!{AtomicAppend,              AtomicAppendOp<CPUContext>}
register_cpu_operator!{CreateTensorVector,        CreateTensorVectorOp<CPUContext>}
register_cpu_operator!{TensorVectorSize,          TensorVectorSizeOp<CPUContext>}
register_cpu_operator!{ConcatTensorVector,        ConcatTensorVectorOp<CPUContext>}
register_cpu_operator!{CollectTensor,             CollectTensorOp<CPUContext>}
register_cpu_operator!{PackRecords,               PackRecordsOp}
register_cpu_operator!{UnPackRecords,             UnPackRecordsOp}
register_cpu_operator!{TrimDataset,               TrimDatasetOp}

should_not_do_gradient!{CreateTreeCursor}
should_not_do_gradient!{ResetCursor}
should_not_do_gradient!{ReadNextBatch}
should_not_do_gradient!{ComputeOffset}
should_not_do_gradient!{ReadRandomBatch}
should_not_do_gradient!{CheckDatasetConsistency}
should_not_do_gradient!{Append}
should_not_do_gradient!{AtomicAppend}
should_not_do_gradient!{CreateTensorVector}
should_not_do_gradient!{TensorVectorSize}
should_not_do_gradient!{ConcatTensorVector}
should_not_do_gradient!{CollectTensor}
should_not_do_gradient!{UnPackRecords}
should_not_do_gradient!{PackRecords}

register_blob_serializer!{
    /*
    (TypeMeta::Id<Box<TreeCursor>>()),
    TreeCursorSerializer
    */
}

register_blob_deserializer!{
    /*
    Box<TreeCursor>, 
    TreeCursorDeserializer
    */
}

register_blob_serializer!{
    /*
    (TypeMeta::Id<Arc<Vec<TensorCPU>>>()),
    SharedTensorVectorPtrSerializer
    */
}

register_blob_deserializer!{
    /*
    Arc<Vec<TensorCPU>>,
    SharedTensorVectorPtrDeserializer
    */
}

