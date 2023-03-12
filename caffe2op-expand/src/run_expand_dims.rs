crate::ix!();

impl<Context> ExpandDimsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            dims_(this->template GetRepeatedArgument<int>("dims")) 

        auto originalSize = dims_.size();
        CAFFE_ENFORCE(originalSize > 0, "Parameter `dims` must be provided.");
        std::sort(dims_.begin(), dims_.end());
        dims_.erase(std::unique(dims_.begin(), dims_.end()), dims_.end());
        if (dims_.size() < originalSize) {
          LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
        }
        CAFFE_ENFORCE(dims_.front() >= 0, "Dimension ids must be non-negative.");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        output->CopyFrom(input, true /*async*/);
        if (dims_.empty()) {
          return true;
        }

        auto newDims = input.sizes().vec();
        CAFFE_ENFORCE_GE(
            input.sizes().size() + dims_.size(),
            dims_.back() + 1,
            "Input needs at least ",
            (1 + dims_.back() - dims_.size()),
            " dimensions given `dims`.");
        for (const auto dim : dims_) {
          newDims.insert(newDims.begin() + dim, 1);
        }
        output->Reshape(newDims);
        return true;
        */
    }
}
