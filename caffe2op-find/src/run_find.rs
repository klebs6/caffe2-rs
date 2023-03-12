crate::ix!();

impl<Context> FindOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            missing_value_(
                this->template GetSingleArgument<int>("missing_value", -1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int, long>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& idx = Input(0);
            auto& needles = Input(1);

            auto* res_indices = Output(0, needles.sizes(), at::dtype<T>());

            const T* idx_data = idx.template data<T>();
            const T* needles_data = needles.template data<T>();
            T* res_data = res_indices->template mutable_data<T>();
            auto idx_size = idx.numel();

            // Use an arbitrary cut-off for when to use brute-force
            // search. For larger needle sizes we first put the
            // index into a map
            if (needles.numel() < 16) {
              // Brute force O(nm)
              for (int i = 0; i < needles.numel(); i++) {
                T x = needles_data[i];
                T res = static_cast<T>(missing_value_);
                for (int j = idx_size - 1; j >= 0; j--) {
                  if (idx_data[j] == x) {
                    res = j;
                    break;
                  }
                }
                res_data[i] = res;
              }
            } else {
              // O(n + m)
              std::unordered_map<T, int> idx_map;
              for (int j = 0; j < idx_size; j++) {
                idx_map[idx_data[j]] = j;
              }
              for (int i = 0; i < needles.numel(); i++) {
                T x = needles_data[i];
                auto it = idx_map.find(x);
                res_data[i] = (it == idx_map.end() ? missing_value_ : it->second);
              }
            }

            return true;
        */
    }
}
