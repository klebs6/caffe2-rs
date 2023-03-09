crate::ix!();

impl<Context> SliceGradientOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            starts_(this->template GetRepeatedArgument<int64_t>("starts")),
            ends_(this->template GetRepeatedArgument<int64_t>("ends")),
            statically_inited_(false)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (InputSize() == 4) {
          return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(1));
        } else {
          return DoRunWithType<int64_t>();
        }
        */
    }
    
    #[inline] pub fn do_run_with_type<SIndex>(&mut self) -> bool {
    
        todo!();
        /*
            auto* gdata = Output(0);
        auto& data = Input(0);

        if (InputSize() == 4) {
          ReinitializeAndCopyFrom(&starts_host_, at::dtype<SIndex>().device(CPU), Input(1));
          ReinitializeAndCopyFrom(&ends_host_, at::dtype<SIndex>().device(CPU), Input(2));

          auto& go = Input(3);

          return SliceImpl<SIndex, Context>(
              nullptr, data, starts_host_, ends_host_, &context_, gdata, &go);
        } else {
          if (!statically_inited_) {
            CAFFE_ENFORCE(HasArgument("starts"));
            CAFFE_ENFORCE(HasArgument("ends"));
            CAFFE_ENFORCE_EQ(starts_.size(), ends_.size());

            ReinitializeTensor(
                &starts_host_, {static_cast<int64_t>(starts_.size())}, at::dtype<SIndex>().device(CPU));
            ReinitializeTensor(
                &ends_host_, {static_cast<int64_t>(ends_.size())}, at::dtype<SIndex>().device(CPU));

            memcpy(
                starts_host_.template mutable_data<SIndex>(),
                starts_.data(),
                sizeof(SIndex) * starts_.size());
            memcpy(
                ends_host_.template mutable_data<SIndex>(),
                ends_.data(),
                sizeof(SIndex) * ends_.size());

            statically_inited_ = true;
          }
          auto& go = Input(1);

          return SliceImpl<SIndex, Context>(
              nullptr, data, starts_host_, ends_host_, &context_, gdata, &go);
        }
        */
    }
}
