crate::ix!();

/**
  | Accuracy takes two inputs- predictions
  | and labels, and returns a float accuracy
  | value for the batch.
  | 
  | Predictions are expected in the form
  | of 2-D tensor containing a batch of scores
  | for various classes, and labels are
  | expected in the form of 1-D tensor containing
  | true label indices of samples in the
  | batch.
  | 
  | If the score for the label index in the
  | predictions is the highest among all
  | classes, it is considered a correct
  | prediction.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AccuracyOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    top_k:   i32,
    phantom: PhantomData<T>,
}

impl<T, Context> AccuracyOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            top_k_(this->template GetSingleArgument<int>("top_k", 1))
        */
    }
}

input_tags!{
    AccuracyOp {
        Prediction,
        Label
    }
}

register_cpu_operator!{
    Accuracy, 
    AccuracyOp<f32, CPUContext>
}

num_inputs!{Accuracy, 2}
num_outputs!{Accuracy, 1}

inputs!{Accuracy, 
    0 => ("predictions", "2-D tensor (Tensor<float>) of size (num_batches x num_classes) containing scores"),
    1 => ("labels", "1-D tensor (Tensor<float>) of size (num_batches) having the indices of true labels")
}

outputs!{Accuracy, 
    0 => ("accuracy", "1-D tensor (Tensor<float>) of size 1 containing accuracy")
}

args!{Accuracy, 
    0 => ("top_k", 
        "Count as correct by comparing the true label to the top k scoring classes 
        (default 1: only compare to the top scoring class i.e. argmax)")
}

scalar_type!{Accuracy, TensorProto::FLOAT}

should_not_do_gradient!{Accuracy}
