crate::ix!();

impl<Context> IndexHashOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            seed_(this->template GetSingleArgument<int64_t>("seed", 0)),
            modulo_(this->template GetSingleArgument<int64_t>("modulo", 0)) 

        CAFFE_ENFORCE_GT(modulo_, 0, "MODULO should be > 0");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& indices = Input(INDICES);

            auto* hashed_indices =
                Output(HASHED_INDICES, indices.sizes(), at::dtype<T>());

            CAFFE_ENFORCE_GE(
                static_cast<int64_t>(T::max),
                modulo_,
                "MODULO shouldn't be larger than the numeric limit of the indices");

            auto N = indices.numel();
            auto* indices_data = indices.template data<T>();
            auto* hashed_indices_data = hashed_indices->template mutable_data<T>();

            for (auto i = 0; i < N; i++) {
              hashed_indices_data[i] = hash(indices_data[i]);
            }

            return true;
        */
    }

    #[inline] pub fn hash<T>(&mut self, id: T) -> T {
        todo!();
        /*
            __ubsan_ignore_signed_int_overflow__

            int8_t* bytes = (int8_t*)&id;
            T hashed = seed_ * 0xDEADBEEF;
            for (int i = 0; i < sizeof(T) / sizeof(int8_t); i++) {
              hashed = hashed * 65537 + bytes[i];
            }
            // We want the result of the modulo to be positive. This works under the
            // assumption that modulo_ > 0 which is enforced in the constructor.
            auto modHashed = hashed % modulo_;
            return modHashed >= 0 ? modHashed : modHashed + modulo_;
        */
    }
}
