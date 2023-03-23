crate::ix!();

enhanced_enum!{
    ConvAlgorithm {
        Auto,
        Winograd,
    }
}

enhanced_enum!{
    FusionType {
        FUSION_UNKNOWN,
        FUSION_CONV_RELU,
        FUSION_CONV_SUM,
        FUSION_CONV_SUM_RELU,
    }
}


