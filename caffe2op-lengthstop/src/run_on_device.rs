crate::ix!();

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
