crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SigmoidCrossEntropyWithLogitsGradientOp<T, Context> {
    storage:          OperatorStorage,
    context:          Context,
    log_D_trick:      bool,
    unjoined_lr_loss: bool,
    phantom:          PhantomData<T>,
}

num_inputs!{SigmoidCrossEntropyWithLogitsGradient, 3}

num_outputs!{SigmoidCrossEntropyWithLogitsGradient, 1}

impl<T,Context> SigmoidCrossEntropyWithLogitsGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            log_D_trick_(
                this->template GetSingleArgument<bool>("log_D_trick", false)),
            unjoined_lr_loss_(
                this->template GetSingleArgument<bool>("unjoined_lr_loss", false))
        */
    }
}
