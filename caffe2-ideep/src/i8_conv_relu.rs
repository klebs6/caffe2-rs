crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_CONV_POOL_BASE_FUNCTIONS]
pub struct IDEEPInt8ConvReluOp {
    base: IDEEPInt8ConvOp,
}

impl IDEEPInt8ConvReluOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPInt8ConvOp(operator_def, ws) 

        CAFFE_ENFORCE(zero_point_ == 0);
        last_input_ = BIAS_OR_INPUT_S;
        attr_ = iattr::fuse_relu();
        fusion_type_ = FUSION_CONV_RELU;
        */
    }
}
