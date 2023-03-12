crate::ix!();

pub struct FusedNBitRowwiseQuantizedToFloatOp<const BIT_RATE: i32, T, ConvertFn> {
    storage:    OperatorStorage,
    context:    CPUContext,
    phantom:    PhantomData<T>,
    phantomCFN: PhantomData<ConvertFn>,
}

input_tags!{
    FusedNBitRowwiseQuantizedToFloatOp {
        DataFusedScaleBias
    }
}

output_tags!{
    FusedNBitRowwiseQuantizedToFloatOp {
        DataFloat
    }
}
