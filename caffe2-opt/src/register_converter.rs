crate::ix!();

register_converter!{Conv, ConvConverter}

register_converter!{ConvTranspose, ConvTransposeConverter}

trivial_converter!{Relu}
register_converter!{Relu, ReluConverter}

trivial_converter!{Sum}
register_converter!{Sum, SumConverter}

trivial_converter!{BatchNormalization}
register_converter!{SpatialBN, BatchNormalizationConverter}

trivial_converter!{Flatten}
register_converter!{Flatten, FlattenConverter}

register_converter!{Clip, ClipConverter}

register_converter!{AveragePool, AveragePoolConverter}

register_converter!{MaxPool, MaxPoolConverter}

register_converter!{Concat, ConcatConverter}

register_converter!{FC, FCConverter}

trivial_converter!{Declare}
register_converter!{Declare, DeclareConverter}

trivial_converter!{Export}
register_converter!{Export, ExportConverter}

register_opt_pass_from_func!{FuseNNPACKConvRelu, fuseNNPACKConvRelu}
register_opt_pass_from_func!{AddNNPACK,          addNNPACK}
