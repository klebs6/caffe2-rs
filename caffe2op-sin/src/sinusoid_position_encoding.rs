crate::ix!();

/**
  | Calculates a sinusoid position encoding
  | tensor as described in https://arxiv.org/abs/1706.03762.
  | Takes a 2-D tensor (of size M x K) of positions
  | as input, the embedding size as an argument,
  | and outputs a position encoding tensor
  | of size (M x K x embedding_size). Here
  | M is typically the max sequence length
  | and K is typically the batch size.
  | 
  | The input tensor must satisfy input[m,
  | 0] == input[m, k] for all k.
  | 
  | Encoded as amplitude * SIN(pos/alpha^(i/embedding_size))
  | if i is even, else amplitude * COS(pos/alpha^(i/embedding_size)).
  | Here, pos is the position, alpha and
  | amplitude are tuning parameters, i
  | is the current dimension for the embedding,
  | and embedding_size is the number of
  | total dimensions in the embedding.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SinusoidPositionEncodingOp<Context> {
    storage:         OperatorStorage,
    context:         Context,
    embedding_size:  i32,
    alpha:           f32,
    amplitude:       f32,
}

register_cpu_operator!{
    SinusoidPositionEncoding,
    SinusoidPositionEncodingOp<CPUContext>
}

num_inputs!{SinusoidPositionEncoding, 1}

num_outputs!{SinusoidPositionEncoding, 1}

inputs!{SinusoidPositionEncoding, 
    0 => ("positions", "2-D tensor of positions to be encoded")
}

outputs!{SinusoidPositionEncoding, 
    0 => ("output", "3-D tensor representing the positional encoding")
}

args!{SinusoidPositionEncoding, 
    0 => ("embedding_size", "Desired embedding size/number of dimensions -- defaults to 100"),
    1 => ("alpha", "Sinusoid tuning parameter -- defaults to 10000"),
    2 => ("amplitude", "Amplitude of Sin/Cos output")
}

impl<Context> SinusoidPositionEncodingOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            embedding_size_( this->template GetSingleArgument<int>("embedding_size", 100)),
            alpha_(this->template GetSingleArgument<float>("alpha", 10000)),
            amplitude_(this->template GetSingleArgument<float>("amplitude", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, this->template Input<Tensor>(0, CPU));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
    
        todo!();
        /*
            auto& positions = Input(0);

        CAFFE_ENFORCE_EQ(positions.dim(), 2, "POSITIONS should be a 2-D tensor");

        auto shape = positions.sizes().vec();
        shape.push_back(embedding_size_);
        auto* output = Output(0, shape, at::dtype<float>());

        int M = shape[0];
        int K = shape[1];
        const Index* idxs = positions.template data<Index>();
        float* out = output->template mutable_data<float>();

        float log_alpha = std::log(alpha_);
        float max_alpha_pow =
            ((float)embedding_size_ - 1.0f) / (float)embedding_size_;

        for (int i = 0; i < M; ++i) {
          float pos = (float)idxs[i * K];

          // Compute the embedding for position i, example 0 first
          float* row = &out[i * K * embedding_size_];
          Eigen::Map<Eigen::VectorXf> row_map(row, embedding_size_, 1);
          auto row_array = row_map.array();

          float log_pos = std::log(pos);
          row_array.setLinSpaced(
              embedding_size_, log_pos, log_pos - log_alpha * max_alpha_pow);
          row_array = row_array.exp().eval();
          // row_array[k] == pos / alpha^(k / embedding_size)

          // Phase shift so that alternating elements are cosines
          for (int k = 1; k < embedding_size_; k += 2) {
            row[k] += (float)M_PI_2;
          }
          row_array = amplitude_ * row_array.sin().eval();

          // Copy the embedding to position i in the other examples
          for (int j = 1; j < K; ++j) {
            int base = i * K * embedding_size_;
            std::copy(
                &out[base],
                &out[base + embedding_size_],
                &out[base + j * embedding_size_]);
          }
        }
        return true;
        */
    }
}
