crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_CONV_POOL_BASE_FUNCTIONS]
pub struct IDEEPChannelShuffleOp<T, Context> {
    base:    IDEEPConvPoolOpBase,
    context: Context,
    phantom: PhantomData<T>,
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

impl<T, Context> IDEEPChannelShuffleOp<T, Context> {

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
