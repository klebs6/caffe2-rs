crate::ix!();

impl<Context> SqueezeOp<Context> {
    
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

        CAFFE_ENFORCE_GT(
            input.dim(),
            dims_.back(),
            "Input needs at least ",
            (dims_.back() + 1),
            " dimensions.");

        std::vector<int> newDims = ComputeDims(input.sizes(), dims_);
        output->Reshape(newDims);
        return true;
        */
    }
    
    #[inline] pub fn compute_dims(input_dims: &[i32], dims: Vec<i32>) -> Vec<i32> {
        
        todo!();
        /*
            size_t j = 0;
        std::vector<int> newDims;
        for (size_t i = 0; i < inputDims.size(); ++i) {
          if (j < dims.size() && dims[j] == i) {
            CAFFE_ENFORCE_EQ(
                inputDims[i],
                1,
                "Dimension ",
                i,
                " of input must be 1",
                " instead of ",
                inputDims[i],
                ".");
            ++j;
            continue;
          }
          newDims.push_back(inputDims.at(i));
        }
        return newDims;
        */
    }
}
