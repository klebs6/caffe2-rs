crate::ix!();

impl<T, Context> FlexibleTopKOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        todo!();
        /*
            auto& input = Input(0);
          auto& k = Input(1);

          const T* input_data = input.template data<T>();
          const int64_t* k_data = k.template data<int64_t>();

          // get flatten shape of input
          CAFFE_ENFORCE_GT(input.dim(), 0);
          vector<int64_t> input_dims = input.sizes().vec();
          vector<int64_t> linear_shape = {
              size_to_dim_(input_dims.size() - 1, input_dims), input_dims.back()};
          CAFFE_ENFORCE_EQ(
              linear_shape[0],
              k.numel(),
              "first n-1 dims of input data and K does not match.");

          int64_t output_size = 0;
          for (int64_t i = 0; i < linear_shape[0]; ++i) {
            CAFFE_ENFORCE(
                linear_shape[1] >= k_data[i],
                "k should not be greater than last dim, error at index ",
                i,
                ", with value: ",
                k_data[i]);
            CAFFE_ENFORCE(
                k_data[i] > 0,
                "k should be greater than 0, error at index ",
                i,
                ",  with value: ",
                k_data[i]);
            output_size += k_data[i];
          }
          auto* values = Output(0, {output_size}, at::dtype<T>());
          auto* indices = Output(1, {output_size}, at::dtype<int64_t>());
          T* values_data = values->template mutable_data<T>();
          int64_t* indices_data = indices->template mutable_data<int64_t>();

          int64_t output_offset = 0;
          // Sort preserving indices
          for (int64_t i = 0; i < linear_shape[0]; ++i) {
            // Build a min-heap, the heap element is pair of (value, idx)
            // the top of the heap is the smallest value
            std::priority_queue<
                std::pair<T, int64_t>,
                std::vector<std::pair<T, int64_t>>,
                ValueCmp<T>>
                PQ;

            int64_t k_ = k_data[i];
            for (int64_t j = 0; j < linear_shape[1]; ++j) {
              const T value = input_data[i * linear_shape[1] + j];
              if (PQ.size() < k_ || value > PQ.top().first) {
                PQ.push(std::make_pair(value, j));
              }
              if (PQ.size() > k_) {
                PQ.pop();
              }
            }
            for (int64_t j = 0; j < k_; ++j) {
              auto& pqElem = PQ.top();
              values_data[output_offset + k_ - j - 1] = pqElem.first;
              indices_data[output_offset + k_ - j - 1] = pqElem.second;
              PQ.pop();
            }
            output_offset += k_;
          }

          return true;
        */
    }
}
