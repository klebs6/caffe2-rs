crate::ix!();

type BufferDataType = (f32, i32);

/**
  | APMeter computes Average Precision
  | for binary or multi-class classification.
  | 
  | It takes two inputs: prediction scores
  | P of size (n_samples x n_classes), and
  | true labels Y of size (n_samples x n_classes).
  | 
  | It returns a single float number per
  | class for the average precision of that
  | class.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct APMeterOp<T, Context> {
    storage:     OperatorStorage,
    context:     Context,

    /// Buffer the predictions for each class
    buffers:     Vec<Vec<BufferDataType>>,

    /// Capacity of the buffer
    buffer_size: i32,

    /// Used buffer
    buffer_used: i32,
    phantom:     PhantomData<T>,
}

impl<T,Context> APMeterOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            buffer_size_(
                this->template GetSingleArgument<int32_t>("buffer_size", 1000)),
            buffer_used_(0)
        */
    }
}

input_tags!{
    APMeterOp {
        Prediction,
        Label
    }
}

register_cpu_operator!{
    APMeter, 
    APMeterOp::<f32, CPUContext>
}

num_inputs!{APMeter, 2}

num_outputs!{APMeter, 1}

inputs!{APMeter, 
    0 => ("predictions", "2-D tensor (Tensor<float>) of size (num_samples x num_classes) containing prediction scores"),
    1 => ("labels", "2-D tensor (Tensor<float>) of size (num_samples) containing true labels for each sample")
}

outputs!{APMeter, 
    0 => ("AP", "1-D tensor (Tensor<float>) of size num_classes containing average precision for each class")
}

args!{APMeter, 
    0 => ("buffer_size", "(int32_t) indicates how many predictions should the op buffer. defaults to 1000")
}

scalar_type!{APMeter, TensorProto::FLOAT}

should_not_do_gradient!{APMeter}
