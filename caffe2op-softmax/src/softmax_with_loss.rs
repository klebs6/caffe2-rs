crate::ix!();

/**
  | Combined Softmax and Cross-Entropy
  | loss operator. The operator first computes
  | the softmax normalized values for each
  | layer in the batch of the given input,
  | then computes cross-entropy loss.
  | 
  | This operator is numerically more stable
  | than separate `Softmax` and `CrossEntropy`
  | ops. The inputs are a 2-D tensor `logits`
  | of size (batch_size x input_feature_dimensions),
  | which represents the unscaled log probabilities,
  | and a 1-dimensional integer `labels`
  | tensor for ground truth.
  | 
  | An optional third input blob (`weight_tensor`)
  | can be used to weight the samples for
  | the loss, which is useful if the training
  | set is unbalanced.
  | 
  | This operator outputs a `softmax` tensor
  | which contains the probability for
  | each label for each example (same shape
  | is `logits` input), and a scalar `loss`
  | value, which is the averaged cross-entropy
  | loss between the softmax probabilities
  | and the ground truth values. Use parameter
  | `label_prob`=1 to enable inputting
  | labels as a probability distribution.
  | 
  | Softmax cross-entropy loss function:
  | 
  | $$loss(x, class) = -\log{\biggl(\frac{\exp(x[class])}{\sum_{j}
  | \exp(x[j])}\biggr)} = -x[class] +
  | \log{\biggl(\sum_{j} \exp(x[j])\biggr)}$$
  | 
  | or if the `weight_tensor` has been passed:
  | 
  | $$loss(x, class) = weight[class]\biggl(-x[class]
  | + \log{\biggl(\sum_{j} \exp(x[j])\biggr)}\biggr)$$
  | 
  | The `logits` input does not need to explicitly
  | be a 2D vector; rather, it will be coerced
  | into one. For an arbitrary n-dimensional
  | tensor `X` in $[a_0, a_1, ..., a_{k-1},
  | a_k, ..., a_{n-1}]$, where k is the `axis`
  | provided, then `X` will be coerced into
  | a 2-dimensional tensor with dimensions
  | $[(a_0 ... * a_{k-1}), (a_k * ... * a_{n-1})]$.
  | For the default case where `axis`=1,
  | the `X` tensor will be coerced into a
  | 2D tensor of dimensions $[a_0, (a_1
  | * ... * a_{n-1})]$, where $a_0$ is often
  | the batch size. In this situation, we
  | must have $a_0 = N$ and $a_1 * ... * a_{n-1}
  | = D$. Each of these dimensions must be
  | matched correctly, or else the operator
  | will throw errors.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_with_loss_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SoftmaxWithLossOp<T,Context> {

    storage:               OperatorStorage,
    context:               Context,

    scale:                 f32,
    label_prob_mode:       i32,
    average_by_batch_size: i32,
    order:                 StorageOrder,
    axis:                  i32,

    /// Per example loss
    losses:                Tensor,

    /// per example row max
    rowmax:                Tensor,

    /// unignored weights
    weights:               Tensor,

    /// Vector of ones for summing via dot prod
    sum_multiplier:        Tensor,
    total_weight_ptr:      Tensor,

    /// passed to a function
    scratch:               Tensor, //default {Context::GetDeviceType()};

    /**
      | Input: X (logits), T (labels);
      | 
      | Output: P (probs), Y
      |
      */
    phantom:               PhantomData<T>,
}

register_cpu_operator!{SoftmaxWithLoss, SoftmaxWithLossOp<f32, CPUContext>}

num_inputs!{SoftmaxWithLoss, (2,3)}

num_outputs!{SoftmaxWithLoss, (2,3)}

inputs!{SoftmaxWithLoss, 
    0 => ("logits",        "*(type: Tensor`<float>`)* Input tensor."),
    1 => ("labels",        "*(type: Tensor`<float>`)* Ground truth label tensor."),
    2 => ("weight_tensor", "*(type: Tensor`<float>`)* [OPTIONAL] Blob used to weight the samples for the loss.")
}

outputs!{SoftmaxWithLoss, 
    0 => ("softmax", "*(type: Tensor`<float>`)* Softmax output tensor."),
    1 => ("loss",    "*(type: float)* Averaged cross-entropy loss output.")
}

args!{SoftmaxWithLoss, 
    0 => ("label_prob", "*(type: int; default: 0)* Setting to 1 enables inputting labels as probability distribution."),
    1 => ("axis",       "*(type: int; default: 1)* Axis of the inputs when coerced to 2D."),
    2 => ("scale",      "*(type: float)* Average loss output scaling factor (must be >= 0)."),
    3 => ("order",      "*(type: string; default: 'NCHW')* Order of blob dimensions (only 'NCHW' is supported currently).")
}

tensor_inference_function!{SoftmaxWithLoss, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          ArgumentHelper helper(def);
          auto axis = helper.GetSingleArgument<int32_t>("axis", 1);

          vector<TensorShape> out(2);

          auto logits = in[0]; // Tensor with Shape [batch_size, num_classes]
          auto labels = in[1]; // Tensor with shape [batch_size, ]
          const auto canonical_axis =
              canonical_axis_index_(axis, logits.dims().size());
          const int batch_size =
              size_to_dim_(canonical_axis, GetDimsVector(logits));
          const int num_classes =
              size_from_dim_(canonical_axis, GetDimsVector(logits));

          out[0].set_data_type(logits.data_type());
          out[0].add_dims(batch_size);
          out[0].add_dims(num_classes);

          return out;
        */
    }
}

impl<T,Context> SoftmaxWithLossOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            scale_(this->template GetSingleArgument<float>("scale", 1.)),
            label_prob_mode_( this->template GetSingleArgument<int>("label_prob", 0)),
            average_by_batch_size_( this->template GetSingleArgument<int>("average_by_batch_size", 0)),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))),
            axis_(this->template GetSingleArgument<int>("axis", 1)) 

        CAFFE_ENFORCE(scale_ >= 0);
        CAFFE_ENFORCE_EQ(
            order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
        */
    }
}
