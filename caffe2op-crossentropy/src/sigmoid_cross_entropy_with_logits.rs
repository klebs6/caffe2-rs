crate::ix!();

/**
  | Given two matrices logits and targets,
  | of same shape, (batch_size, num_classes),
  | computes the sigmoid cross entropy
  | between the two.
  | 
  | Returns a tensor of shape (batch_size,)
  | of losses for each example.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SigmoidCrossEntropyWithLogitsOp<T, Context> {
    storage:          OperatorStorage,
    context:          Context,
    log_D_trick:      bool,
    unjoined_lr_loss: bool,
    phantom:          PhantomData<T>,
}

num_inputs!{SigmoidCrossEntropyWithLogits, 2}

num_outputs!{SigmoidCrossEntropyWithLogits, 1}

inputs!{SigmoidCrossEntropyWithLogits, 
    0 => ("logits",  "matrix of logits for each example and class."),
    1 => ("targets", "matrix of targets, same shape as logits.")
}

outputs!{SigmoidCrossEntropyWithLogits, 
    0 => ("xentropy", "Vector with the total xentropy for each example.")
}

args!{SigmoidCrossEntropyWithLogits, 
    0 => ("log_D_trick",      "default is false; if enabled, will use the log d trick to avoid the vanishing gradients early on; see Goodfellow et. al (2014)"),
    1 => ("unjoined_lr_loss", "default is false; if enabled, the model will be allowed to train on an unjoined dataset, where some examples might be false negative and might appear in the dataset later as (true) positive example.")
}

identical_type_and_shape_of_input_dim!{SigmoidCrossEntropyWithLogits, (0, 0)}

impl<T,Context> SigmoidCrossEntropyWithLogitsOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            log_D_trick_(
                this->template GetSingleArgument<bool>("log_D_trick", false)),
            unjoined_lr_loss_(
                this->template GetSingleArgument<bool>("unjoined_lr_loss", false)) 

        CAFFE_ENFORCE(
            !(log_D_trick_ && unjoined_lr_loss_),
            "log_D_trick_ and unjoined_lr_loss_ cannot be set as True simultaneously");
        */
    }
}
