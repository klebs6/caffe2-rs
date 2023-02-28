crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
};

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
pub struct AccuracyOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;

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

impl AccuracyOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(PREDICTION);
      auto& label = Input(LABEL);

      CAFFE_ENFORCE_EQ(X.dim(), 2);
      int N = X.dim32(0);
      int D = X.dim32(1);
      CAFFE_ENFORCE_EQ(label.dim(), 1);
      CAFFE_ENFORCE_EQ(label.dim32(0), N);
      auto* Y = Output(0, vector<int64_t>(), at::dtype<float>());
      const auto* Xdata = X.data<float>();
      const auto* labelData = label.data<int>();
      const int top_k = top_k_;
      int correct = 0;

      // it's equivalent to using a stable sorting algorithm to sort the
      // classes (with their predictions as key) and then check whether
      // the label is within the first top_k slots.
      for (int i = 0; i < N; ++i) {
        auto label_i = labelData[i];
        auto label_pred = Xdata[i * D + label_i];
        int ngt = 1;
        for (int j = 0; j < D; ++j) {
          auto pred = Xdata[i * D + j];
          if ((pred > label_pred) || (pred == label_pred && j < label_i)) {
            if (++ngt > top_k) {
              break;
            }
          }
        }
        if (ngt <= top_k) {
          ++correct;
        }
      }
      CAFFE_ENFORCE_LE(correct, N);
      *(Y->template mutable_data<float>()) = static_cast<float>(correct) / N;

      return true;
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
