crate::ix!();

/**
  | Respectively compute accuracy score
  | for each class given a number of instances
  | and predicted scores of each class for
  | each instance.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MultiClassAccuracyOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

register_cpu_operator!{
    MultiClassAccuracy, 
    MultiClassAccuracyOp<float, CPUContext>
}

num_inputs!{MultiClassAccuracy, 2}

num_outputs!{MultiClassAccuracy, 2}

inputs!{MultiClassAccuracy, 
    0 => ("prediction", "2-D float tensor (N,D,) of predicted scores of each class for each data. N is the number of instances, i.e., batch size. D is number of possible classes/labels."),
    1 => ("labels",     "1-D int tensor (N,) of labels for each instance.")
}

outputs!{MultiClassAccuracy, 
    0 => ("accuracies", "1-D float tensor (D,) of accuracy for each class. If a class has no instance in the batch, its accuracy score is set to zero."),
    1 => ("amounts",    "1-D int tensor (D,) of number of instances for each class in the batch.")
}

should_not_do_gradient!{MultiClassAccuracy}

input_tags!{
    MultiClassAccuracyOp {
        Prediction,
        Label
    }
}
