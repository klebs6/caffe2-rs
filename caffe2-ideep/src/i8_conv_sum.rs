crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_CONV_POOL_BASE_FUNCTIONS]
pub struct IDEEPInt8ConvSumOp {
    base: IDEEPInt8ConvOp,
}

num_inputs!{Int8ConvSum, (2,4)}

num_outputs!{Int8ConvSum, 1}

tensor_inference_function!{Int8ConvSum, /* ConvPoolOpBase<CPUContext>::TensorInferenceForConv */}

cost_inference_function!{Int8ConvSum, /* OpSchema::CostInferenceFunctionType( ConvPoolOpBase<CPUContext>::CostInferenceForConv) */ }

allow_inplace!{Int8ConvSum, vec![(2, 0), (3, 0)]}

impl IDEEPInt8ConvSumOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPInt8ConvOp(operator_def, ws) 
        last_input_ = INPUT_S;
        attr_ = iattr::fuse_sum();
        fusion_type_ = FUSION_CONV_SUM;
        */
    }
}
