crate::ix!();

use crate::{
    OperatorDef,
    IDEEPConvPoolOpBase,
    Workspace
};

pub struct IDEEPChannelShuffleOp<T, Context> {
    base:    IDEEPConvPoolOpBase,
    context: Context,
    phantom: PhantomData<T>,
}

impl<T, Context> IDEEPChannelShuffleOp<T, Context> {

    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
        Self {
            base: IDEEPConvPoolOpBase::new(operator_def, ws),
        }
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();

        /*
            const auto& X = Input(INPUT);
        auto* Y = Output(OUTPUT);

        ideep::channel_shuffle_forward::compute(X, *Y, group_);

        return true;
        */
    }
}

input_tags!{
    IDEEPChannelShuffleOp  
    { 
        Input 
    }
}

output_tags!{
    IDEEPChannelShuffleOp 
    {
        Output
    }
}

pub struct ChannelShuffleGradientOp {
    base: IDEEPConvPoolOpBase,
}

impl ChannelShuffleGradientOp {

    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

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

register_ideep_operator!{ChannelShuffle, IDEEPChannelShuffleOp}

register_ideep_operator!{ChannelShuffleGradient, ChannelShuffleGradientOp}
