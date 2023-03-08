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

    storage: OperatorStorage,
    context: Context,

    /// Buffer the predictions for each class
    buffers: Vec<Vec<BufferDataType>>,

    /// Capacity of the buffer
    buffer_size: i32,

    /// Used buffer
    buffer_used: i32,
    phantom: PhantomData<T>,
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

impl APMeterOp<f32, CPUContext> {
    
    /// Buffer predictions for N sample and D classes
    #[inline] pub fn buffer_predictions(
        &mut self, 
        xdata:       *const f32,
        label_data:  *const i32,
        n:           i32,
        d:           i32)  
    {
        todo!();
        /*
           if (buffers_.empty()) {
            // Initialize the buffer
            buffers_.resize(D, std::vector<BufferDataType>(buffer_size_));
          }
          DCHECK_EQ(buffers_.size(), D);

          // Fill atmose buffer_size_ data at a time, so truncate the input if needed
          if (N > buffer_size_) {
            XData = XData + (N - buffer_size_) * D;
            labelData = labelData + (N - buffer_size_) * D;
            N = buffer_size_;
          }

          // Reclaim space if not enough space in the buffer to hold new data
          int space_to_reclaim = buffer_used_ + N - buffer_size_;
          if (space_to_reclaim > 0) {
            for (auto& buffer : buffers_) {
              std::rotate(
                  buffer.begin(), buffer.begin() + space_to_reclaim, buffer.end());
            }
            buffer_used_ -= space_to_reclaim;
          }

          // Fill the buffer
          for (int i = 0; i < D; i++) {
            for (int j = 0; j < N; j++) {
              buffers_[i][buffer_used_ + j].first = XData[j * D + i];
              buffers_[i][buffer_used_ + j].second = labelData[j * D + i];
            }
          }

          buffer_used_ += N;
               
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(PREDICTION);
      auto& label = Input(LABEL);

      // Check dimensions
      DCHECK_EQ(X.dim(), 2);
      int N = X.dim32(0);
      int D = X.dim32(1);
      DCHECK_EQ(label.dim(), 2);
      DCHECK_EQ(label.dim32(0), N);
      DCHECK_EQ(label.dim32(1), D);
      auto* Y = Output(0, {D}, at::dtype<float>());

      const auto* Xdata = X.data<float>();
      const auto* labelData = label.data<int>();
      auto* Ydata = Y->template mutable_data<float>();

      BufferPredictions(Xdata, labelData, N, D);

      // Calculate AP for each class
      for (int i = 0; i < D; i++) {
        auto& buffer = buffers_[i];
        // Sort predictions by score
        std::stable_sort(
            buffer.begin(),
            buffer.begin() + buffer_used_,
            [](const BufferDataType& p1, const BufferDataType& p2) {
              return p1.first > p2.first;
            });
        // Calculate cumulative precision for each sample
        float tp_sum = 0.0;
        float precision_sum = 0.0;
        int ntruth = 0;
        for (int j = 0; j < buffer_used_; j++) {
          tp_sum += buffer[j].second;
          if (buffer[j].second == 1) {
            ntruth += 1;
            precision_sum += tp_sum / (j + 1);
          }
        }

        // Calculate AP
        Ydata[i] = precision_sum / std::max(1, ntruth);
      }

      return true;
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
