crate::ix!();

impl<Context> AssertOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            error_msg_(
                this->template GetSingleArgument<std::string>("error_msg", ""))
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            // Copy into CPU context for comparison
            cmp_tensor_.CopyFrom(Input(0));
            auto* cmp_data = cmp_tensor_.template data<T>();

            for (int64_t i = 0; i < cmp_tensor_.numel(); ++i) {
              CAFFE_ENFORCE((bool)cmp_data[i], [&]() {
                std::stringstream ss;
                ss << "Assert failed for element " << i
                   << " in tensor, value: " << cmp_data[i] << "\n";
                if (!error_msg_.empty()) {
                  ss << "Error message: " << error_msg_;
                }
                return ss.str();
              }());
            }
            return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<long, int, bool>>::call(this, Input(0));
        */
    }
}
