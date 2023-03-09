crate::ix!();

register_cpu_operator!{AddPadding,        AddPaddingOp<CPUContext>}
register_cpu_operator!{RemovePadding,     RemovePaddingOp<CPUContext>}
register_cpu_operator!{GatherPadding,     GatherPaddingOp<CPUContext>}
register_cpu_operator!{PadEmptySamples,   PadEmptySamplesOp<CPUContext>}
