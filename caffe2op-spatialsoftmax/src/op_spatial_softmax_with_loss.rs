crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorDef,
    StorageOrder,
    CPUContext,
    Tensor,
    OperatorStorage,
};

/**
  | Combined Spatial Softmax and Cross-Entropy
  | loss operator.
  | 
  | Similar to SoftmaxWithLoss, this operator
  | computes the spatial softmax normalized
  | values for each layer in the batch of
  | the given input, after which cross-entropy
  | loss is computed.
  | 
  | This operator is numerically more stable
  | than separate Softmax and CrossEntropy
  | ops.
  | 
  | The inputs are a 2-D tensor (Tensor)
  | of size (batch_size x input_feature_dimensions)
  | and tensor of labels (ground truth).
  | 
  | Output is tensor with the probability
  | for each label in a pixel for each example
  | (N x D x W x H) and averaged loss (scalar).
  | 
  | For spatial softmax, weighting is by
  | x,y position of the input.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SpatialSoftmaxWithLossOp<T,Context> {

    storage:           OperatorStorage,
    context:           Context,

    scale:             f32,
    order:             StorageOrder,

    /// Per example loss
    losses:            Tensor,

    /// per example row max
    rowmax:            Tensor,

    /// unignored weights
    weights:           Tensor,

    /// Vector of ones for summing via dot prod
    sum_multiplier:    Tensor,

    total_weight_ptr:  Tensor,

    /// {Context::GetDeviceType()};
    scratch:           Tensor,

    /**
      | Input: X (logits), T (labels);
      | 
      | Output: P (probs), Y
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{SpatialSoftmaxWithLoss, (2,3)}

num_outputs!{SpatialSoftmaxWithLoss, 2}

inputs!{SpatialSoftmaxWithLoss, 
    0 => ("logits",        "Unscaled log probabilities"),
    1 => ("labels",        "Ground truth"),
    2 => ("weight_tensor", "Optional blob to be used to weight the samples for the loss. With spatial set, weighting is by x,y of the input")
}

outputs!{SpatialSoftmaxWithLoss, 
    0 => ("softmax", "Tensor with softmax cross entropy loss"),
    1 => ("loss",    "Average loss")
}

tensor_inference_function!{SpatialSoftmaxWithLoss, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          ArgumentHelper helper(def);
          vector<TensorShape> out(2);

          auto logits = in[0]; // Tensor with Shape [batch_size, num_classes]
          auto labels = in[1]; // Tensor with shape [batch_size, ]
          auto batch_size = logits.dims().Get(0);
          auto num_classes = logits.dims().Get(1);

          CAFFE_ENFORCE_EQ(logits.dims_size(), 4);
          CAFFE_ENFORCE_EQ(labels.dims_size(), 3);
          out[0].set_data_type(logits.data_type());
          out[0].add_dims(batch_size);
          out[0].add_dims(num_classes);
          out[0].add_dims(in[0].dims(2));
          out[0].add_dims(in[0].dims(3));
          // Output 2 is scalar shape, so no dims added
          return out;
        */
    }
}

impl<T,Context> SpatialSoftmaxWithLossOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            scale_(this->template GetSingleArgument<float>("scale", 1.)),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))) 

        CAFFE_ENFORCE(scale_ >= 0);
        CAFFE_ENFORCE_EQ(
            order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
        */
    }
}

///--------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SpatialSoftmaxWithLossGradientOp<T,Context> {

    storage:           OperatorStorage,
    context:           Context,

    scale:             f32,
    sum_multiplier:    Tensor,

    /// unignored weights
    weights:           Tensor,
    total_weight_ptr:  Tensor,
    order:             StorageOrder,
    only_loss:         bool,

    /// {Context::GetDeviceType()};
    scratch:           Tensor,

    /**
      | Input: X, T, P, dY;
      | 
      | Output: dX
      |
      */
    phantom:           PhantomData<T>,
}

num_outputs!{SpatialSoftmaxWithLossGradient, 1}

impl<T,Context> SpatialSoftmaxWithLossGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            scale_(this->template GetSingleArgument<float>("scale", 1.)),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))),
            only_loss_(this->template GetSingleArgument<bool>("only_loss", false)) 

        CAFFE_ENFORCE(scale_ >= 0);
        CAFFE_ENFORCE_EQ(
            order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
        */
    }
}

register_cpu_operator!{
    SpatialSoftmaxWithLoss,
    SpatialSoftmaxWithLossOp<f32, CPUContext>}

register_cpu_operator!{
    SpatialSoftmaxWithLossGradient,
    SpatialSoftmaxWithLossGradientOp<f32, CPUContext>}

pub const DONT_CARE: i32 = -1;

impl<T,Context> SpatialSoftmaxWithLossOp<T,Context> {

    #[inline] pub fn run_on_device_f32_cpu_context(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0); // Logits
      auto& T = Input(1); // Labels / targets

      int N, D;
      N = X.dim32(0);
      D = X.dim32(1);
      auto* P =
          Output(0, X.sizes(), at::dtype<float>()); // Probabilities from softmax

      if (!sum_multiplier_.defined()) {
        sum_multiplier_ = caffe2::empty({D}, at::dtype<float>().device(CPU));
        math::Set<float, CPUContext>(
            D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
      } else if (sum_multiplier_.numel() != D) {
        sum_multiplier_.Resize(D);
        math::Set<float, CPUContext>(
            D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
      }

      float* Pdata = P->template mutable_data<float>();
      const float* weights = (InputSize() > 2 ? Input(2).data<float>() : nullptr);
      CAFFE_ENFORCE_EQ(X.dim(), 4);
      CAFFE_ENFORCE_EQ(T.dim(), 3);
      CAFFE_ENFORCE_EQ(T.dim32(0), N);

      int H = X.dim32(2);
      int W = X.dim32(3);

      const float* Xdata = X.data<float>();

      for (int i = 0; i < N; ++i) {
        for (int y = 0; y < H; ++y) {
          for (int x = 0; x < W; ++x) {
            // Subtract max on each cell for numerical reasons
            float max_val = (-1e20f);
            for (int c = 0; c < D; ++c) {
              // TODO optimize
              int idx = i * (H * W * D) + c * (H * W) + y * W + x;
              max_val = std::max(max_val, Xdata[idx]);
            }

            // Exponentiate
            float expsum = 0.0f;
            for (int c = 0; c < D; ++c) {
              int idx = i * (H * W * D) + c * (H * W) + y * W + x;
              float expx = exp(Xdata[idx] - max_val);
              Pdata[idx] = expx;
              expsum += expx;
            }

            // Normalize
            for (int c = 0; c < D; ++c) {
              int idx = i * (H * W * D) + c * (H * W) + y * W + x;
              Pdata[idx] /= expsum;
            }
          }
        }
      }

      // Compute the avg cross-entropy loss
      auto* avg_loss =
          Output(1, vector<int64_t>(), at::dtype<float>()); // Average loss
      float* avg_loss_data = avg_loss->template mutable_data<float>();
      const int* label_data = T.data<int>();

      float sum_label_xent = 0.0f;
      float total_weight = 0.0;

      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          for (int i = 0; i < N; i++) {
            int label_idx = i * H * W + y * W + x;
            int label = label_data[label_idx];
            if (label != DONT_CARE) {
              CAFFE_ENFORCE(
                  label < D && label >= 0,
                  "Label seems incorrect:label value larger than number of classes",
                  label_data[i],
                  " vs ",
                  D);
              int idx = i * (H * W * D) + label * (H * W) + y * W + x;
              float w = weights ? weights[label_idx] : 1.0;
              total_weight += w;
              sum_label_xent += -log(std::max(Pdata[idx], 1e-20f)) * w;
            }
          }
        }
      }
      if (total_weight != 0.0) {
        *avg_loss_data = sum_label_xent / total_weight;
      } else {
        *avg_loss_data = 0.0;
      }
      return true;
        */
    }
}

impl<T,Context> SpatialSoftmaxWithLossGradientOp<T, Context> {

    #[inline] pub fn run_on_device_f32_cpu_context(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0); // Logits
      auto& T = Input(1); // Labels / targets
      // Input(2) is weights if given
      auto& P = Input(InputSize() - 2); // Probabilities from softmax
      auto& d_avg_loss = Input(InputSize() - 1); // Gradient w.r.t. avg loss

      const float* weights = (InputSize() > 4 ? Input(2).data<float>() : nullptr);
      int N, D;
      N = X.dim32(0);
      D = X.dim32(1);
      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      CAFFE_ENFORCE_EQ(T.dim32(0), N);
      CAFFE_ENFORCE_EQ(X.dim(), 4);
      CAFFE_ENFORCE_EQ(T.dim(), 3);

      int H = X.dim32(2);
      int W = X.dim32(3);

      const float* Pdata = P.data<float>();
      float* dX_data = dX->template mutable_data<float>();
      const int* label_data = T.data<int>();

      // Copy softmax probabilities into dX. All but the neuron
      // corresponding to the correct label has gradient equaling e(x_j)
      // which is the probability under softmax.
      context_.CopyFromCPU<float>(P.numel(), Pdata, dX_data);

      float total_weight = 0.0f;
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          for (int i = 0; i < N; ++i) {
            int label_idx = i * H * W + y * W + x;
            int label = label_data[label_idx];

            if (label != DONT_CARE) {
              int idx = i * (H * W * D) + label * (H * W) + y * W + x;

              dX_data[idx] = (dX_data[idx] - 1.0);

              if (weights != nullptr) {
                float weight = weights[label_idx];
                for (int c = 0; c < D; ++c) {
                  int k = i * (H * W * D) + c * (H * W) + y * W + x;
                  dX_data[k] *= weight;
                }
                total_weight += weight;
              } else {
                total_weight += 1.0;
              }
            } else {
              // Set gradient to zero for coordinates where we have dont care
              for (int c = 0; c < D; ++c) {
                int idx = i * (H * W * D) + c * (H * W) + y * W + x;
                dX_data[idx] = 0;
              }
            }
          }
        }
      }

      if (total_weight > 0) {
        math::Scale<float, float, CPUContext>(
            dX->numel(),
            scale_ / total_weight,
            dX->data<float>(),
            dX_data,
            &context_);
      }
      math::Scale<float, float, CPUContext>(
          dX->numel(),
          d_avg_loss.data<float>(),
          dX->data<float>(),
          dX->template mutable_data<float>(),
          &context_);
      return true;
        */
    }
}

pub struct GetSoftmaxWithLossGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSoftmaxWithLossGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> blob_names{
            {I(0), I(1), O(0), GO(1)},
        };

        // Add weight blob, if given
        if (def_.input_size() == 3) {
          blob_names.emplace(blob_names.begin() + 2, I(2));
        }
        return SingleGradientDef(
            "SpatialSoftmaxWithLossGradient",
            "",
            blob_names,
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{SpatialSoftmaxWithLoss, GetSoftmaxWithLossGradient}
