crate::ix!();

/**
  | Same as FC, but weight matrix is supposed
  | to be already pretransposed.
  | 
  | FCTransposed stands for calling blass
  | with no noTrans, noTrans
  |
  */
register_cpu_operator!{
    FCTransposed,
    FullyConnectedOp<
        CPUContext,
        DefaultEngine,
        false /* don't transpose weight */>
}

register_cpu_gradient_operator!{
    FCTransposedGradient,
    FullyConnectedGradientOp<
        CPUContext,
        DefaultEngine,
        DontTransposeWeight>
}
