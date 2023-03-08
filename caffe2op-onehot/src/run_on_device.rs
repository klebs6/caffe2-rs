crate::ix!();

impl<Context> OneHotOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& indices = Input(0);
        CAFFE_ENFORCE_EQ(
            indices.dim(),
            1,
            "indices input must be 1D tensor of data type int64_t");

        // Index size input must be in CPU context
        auto& index_size_tensor = this->template Input<Tensor>(1, CPU);
        CAFFE_ENFORCE_EQ(
            index_size_tensor.numel(),
            1,
            "index_size_tensor input must be scalar of data type int64_t");

        auto batch_size = indices.numel();
        auto index_size = *index_size_tensor.template data<int64_t>();
        auto one_hots = Output(0);
        one_hots->Resize(batch_size, index_size);
        auto output_size = one_hots->numel();
        if (output_size == 0) {
          return true;
        }

        DoOneHotOp(batch_size, index_size, indices, one_hots);
        return true;
        */
    }
}
