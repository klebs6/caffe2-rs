crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_CONV_POOL_BASE_FUNCTIONS]
pub struct ChannelShuffleGradientOp {
    base: IDEEPConvPoolOpBase,
}

input_tags!{
    ChannelShuffleGradientOp  
    {
        OutputGrad
    }
}

output_tags!{
    ChannelShuffleGradientOp 
    {
        InputGrad
    }
}

impl ChannelShuffleGradientOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        Self {
            base: IDEEPConvPoolOpBase::new(operator_def, ws),
        }
    }
    
    #[inline] pub fn run_on_device_with_order_nchw(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dY = Input(OUTPUT_GRAD);
        auto* dX = Output(INPUT_GRAD);

        ideep::channel_shuffle_backward::compute(dY, *dX, group_);

        return true;
        */
    }
}
