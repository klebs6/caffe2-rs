crate::ix!();

impl<T, Context> GluOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            dim_(this->template GetSingleArgument<int>("dim", -1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        vector<int64_t> Yshape;
        Yshape.insert(Yshape.end(), X.sizes().begin(), X.sizes().end());
        const int split_index = dim_ == -1 ? Yshape.size() - 1 : dim_;
        CAFFE_ENFORCE(
            Yshape[split_index] % 2 == 0,
            "Split dimension ",
            Yshape[split_index],
            " should be divided by two");
        const int split_dim_size = Yshape[split_index] / 2;
        const int M = X.size_to_dim(split_index);
        const int N = X.size_from_dim(split_index + 1);
        Yshape[split_index] = split_dim_size;
        auto* Y = Output(0, Yshape, at::dtype<T>());
        ComputeGlu(
            M,
            split_dim_size,
            N,
            X.template data<T>(),
            Y->template mutable_data<T>());
        return true;
        */
    }
}
