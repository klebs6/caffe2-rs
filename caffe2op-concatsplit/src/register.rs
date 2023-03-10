crate::ix!();

register_cpu_operator!{
    Split, 
    SplitOp<CPUContext>
}

register_cpu_operator!{
    SplitByLengths, 
    SplitByLengthsOp<CPUContext>
}

register_cpu_operator!{
    Concat, 
    ConcatOp<CPUContext>
}
