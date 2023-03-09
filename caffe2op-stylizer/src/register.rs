crate::ix!();

register_cpu_operator!{
    PackedInt8BGRANHWCToNCHWCStylizerPreprocess,
    PackedInt8BGRANHWCToNCHWCStylizerPreprocessOp
}

num_inputs!{PackedInt8BGRANHWCToNCHWCStylizerPreprocess, 2}

num_outputs!{PackedInt8BGRANHWCToNCHWCStylizerPreprocess, 1}

///--------------------
register_cpu_operator!{
    BRGNCHWCToPackedInt8BGRAStylizerDeprocess,
    BRGNCHWCToPackedInt8BGRAStylizerDeprocessOp
}

num_inputs!{BRGNCHWCToPackedInt8BGRAStylizerDeprocess, 2}

num_outputs!{BRGNCHWCToPackedInt8BGRAStylizerDeprocess, 1}

#[cfg(caffe2_use_mkldnn)]
register_ideep_operator!{
    BRGNCHWCToPackedInt8BGRAStylizerDeprocess,
    IDEEPFallbackOp<BRGNCHWCToPackedInt8BGRAStylizerDeprocessOp, SkipIndices<0>>
}

#[cfg(caffe2_use_mkldnn)]
register_ideep_operator!{
    PackedInt8BGRANHWCToNCHWCStylizerPreprocess,
    IDEEPFallbackOp<PackedInt8BGRANHWCToNCHWCStylizerPreprocessOp>
}
