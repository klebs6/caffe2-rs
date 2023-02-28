crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorDef,
    OperatorStorage,
};

/**
  | Apply TopK to each segment of the input
  | tensor, where segments are defined
  | by their LENGTHS, and concatenate them
  | in an output tensor of shape=(SIZE(LENGTHs),
  | k).
  | 
  | In case there's less than k values in
  | a segment, the output value will be padded
  | by 0, and the corresponding output indices
  | will be padded by -1.
  |
  */
pub struct LengthsTopKOp<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    k:       i32,
    phantom: PhantomData<T>,
}

register_cpu_operator!{LengthsTopK, LengthsTopKOp<float, CPUContext>}

num_inputs!{LengthsTopK, 2}

num_outputs!{LengthsTopK, 2}

inputs!{LengthsTopK, 
    0 => ("DATA",        "Tensor of rank 1. First dimension must be equal to the sum of lengths"),
    1 => ("LENGTHS",     "Tensor of int32 lengths of rank 1")
}

outputs!{LengthsTopK, 
    0 => ("TopKValue",   "Output top k elements for each segment, with shape=(SIZE(lengths), k)"),
    1 => ("TopKIndices", "Output indices in DATA corresponding to value in TopKValue")
}

args!{LengthsTopK, 
    0 => ("k", "the number of top values to return for each segment, if the number of values is smaller than k, the values would be padded with 0 and indices would be padded with -1.")
}

input_tags!{
    LengthsTopKOp {
        XIn,
        YIn
    }
}

output_tags!{
    LengthsTopKOp {
        TopkValuesOut,
        TopkIndicesOut
    }
}

impl<T,Context> LengthsTopKOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "k", k_, -1) 

        CAFFE_ENFORCE_GE(k_, 1, "k argument must be >= 1");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(X_IN);
      auto& Y = Input(Y_IN);
      int N = Y.dim32(0);
      const T* X_data = X.template data<T>();
      const int* input_len = Y.template data<int>();

      auto output_dims = std::vector<int64_t>({N, k_});
      auto* output_topk_values = Output(TOPK_VALUES_OUT, output_dims, at::dtype<T>());
      auto* output_topk_indices =
          Output(TOPK_INDICES_OUT, output_dims, at::dtype<int>());
      T* output_topk_values_data = output_topk_values->template mutable_data<T>();
      int* output_topk_indices_data =
          output_topk_indices->template mutable_data<int>();

      auto cmp = [](std::pair<T, int64_t>& lhs, std::pair<T, int64_t>& rhs) {
        return lhs.first > rhs.first ||
            (lhs.first == rhs.first && lhs.second < rhs.second);
      };

      // Sort preserving indices
      int next_index = 0;
      for (int64_t i = 0; i < N; ++i) {
        // Build a min-heap, the heap element is pair of (value, idx)
        // the top of the heap is the smallest value
        std::priority_queue<
            std::pair<T, int64_t>,
            std::vector<std::pair<T, int64_t>>,
            decltype(cmp)>
            p_queue(cmp);

        // Maintain the size of heap to be less or equal to k_, so the
        // heap will hold the k_ largest values
        for (int64_t j = 0; j < input_len[i]; ++j) {
          const auto value = X_data[next_index++];
          if (p_queue.size() < k_ || value > p_queue.top().first) {
            p_queue.push(std::make_pair(value, j));
          }
          if (p_queue.size() > k_) {
            p_queue.pop();
          }
        }

        int last_index = p_queue.size();
        for (int64_t j = 0; j < k_; ++j) {
          if (p_queue.size() > 0) {
            auto& pqElem = p_queue.top();
            output_topk_values_data[i * k_ + last_index - j - 1] = pqElem.first;
            output_topk_indices_data[i * k_ + last_index - j - 1] = pqElem.second;
            p_queue.pop();
          } else {
            output_topk_values_data[i * k_ + j] = 0;
            output_topk_indices_data[i * k_ + j] = -1;
          }
        }
      }

      return true;
        */
    }
}

///------------------------------
pub struct LengthsTopKGradientOp<T,Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    k:       i32,
    phantom: PhantomData<T>,
}

register_cpu_operator!{LengthsTopKGradient, LengthsTopKGradientOp<float, CPUContext>}

num_inputs!{LengthsTopKGradient, 3}

num_outputs!{LengthsTopKGradient, 1}

input_tags!{
    LengthsTopKGradientOp {
        LengthIn,
        IndicesIn,
        DerTopkIn
    }
}

output_tags!{
    LengthsTopKGradientOp {
        DerXOut
    }
}

impl<T,Context> LengthsTopKGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "k", k_, -1) 

        CAFFE_ENFORCE_GE(k_, 1, "k argument must be >= 1");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input_len = Input(LENGTH_IN);
      int N = input_len.numel();
      auto& input_indices = Input(INDICES_IN);
      CAFFE_ENFORCE_GE(input_indices.dim(), 2, "input dim must be >= 2");
      CAFFE_ENFORCE_EQ(
          input_indices.numel(), N * k_, "input_indices shape is not correct");
      auto& input_topk = Input(DER_TOPK_IN);
      CAFFE_ENFORCE_EQ(
          input_topk.numel(), N * k_, "input_topk shape is not correct");

      const int* input_len_data = input_len.template data<int>();
      const int* input_indices_data = input_indices.template data<int>();
      const T* input_topk_data = input_topk.template data<T>();

      int num_indices = 0;
      for (int i = 0; i < N; i++) {
        num_indices += input_len_data[i];
      }
      auto* X_out = Output(DER_X_OUT, {num_indices}, at::dtype<T>());
      T* X_out_data = X_out->template mutable_data<T>();
      math::Set<T, Context>(num_indices, 0.0, X_out_data, &context_);

      int index_offset = 0;
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < std::min(input_len_data[i], k_); j++) {
          int cur_index = index_offset + input_indices_data[i * k_ + j];
          CAFFE_ENFORCE_LT(
              cur_index, num_indices, "cur_index should be less than num_indices");
          X_out_data[cur_index] = input_topk_data[i * k_ + j];
        }
        index_offset += input_len_data[i];
      }

      return true;
        */
    }
}

pub struct GetLengthsTopKGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLengthsTopKGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "LengthsTopKGradient",
            "",
            vector<string>{I(1), O(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{LengthsTopK, GetLengthsTopKGradient}
