crate::ix!();

impl UniqueOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& inputTensor = Input(0);
          // use dim32 to enforce that it's fine to have remapping of type int
          int N = inputTensor.dim32(0);
          CAFFE_ENFORCE_EQ(inputTensor.dim(), 1, "Input should be a vector");

          int* remapping = nullptr;
          if (REMAPPING < OutputSize()) {
            auto* remappingTensor =
                Output(REMAPPING, inputTensor.sizes(), at::dtype<int>());
            remapping = remappingTensor->template mutable_data<int>();
          }

          const T* input = inputTensor.template data<T>();
          // TODO(dzhulgakov): if perf becomes an issue consider doing hash table
          // instead of sorting
          order_.resize(N);
          std::iota(order_.begin(), order_.end(), 0);
          std::sort(order_.begin(), order_.end(), [input](const int x, const int y) {
            return input[x] < input[y];
          });
          int K = N;
          for (int i = 1; i < N; ++i) {
            K -= input[order_[i]] == input[order_[i - 1]];
          }
          auto* uniqueTensor = Output(UNIQUE, {K}, at::dtype<T>());
          T* unique = uniqueTensor->template mutable_data<T>();
          K = 0;
          T prev = -1;
          for (int i = 0; i < N; ++i) {
            if (i == 0 || prev != input[order_[i]]) {
              prev = unique[K++] = input[order_[i]];
            }
            if (remapping) {
              remapping[order_[i]] = K - 1;
            }
          }
          return true;
        */
    }
}

register_cpu_operator!{Unique, UniqueOp<CPUContext>}
