crate::ix!();


pub trait FallbackOp {
    type Fallback;
}

pub struct IDEEPExpandDimsOp {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,

    dims:     Vec<i32>,
    fallback: <Self as FallbackOp>::Fallback,
}

impl FallbackOp for IDEEPExpandDimsOp {
    type Fallback = IDEEPFallbackOp<ExpandDimsOp<CPUContext>, dyn SkipIndices<0>>;
}

input_tags!{
    IDEEPExpandDimsOp 
    {
        Input
    }
}

output_tags!{
    IDEEPExpandDimsOp {
        Output
    }
}

impl IDEEPExpandDimsOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            fallback_(operator_def, ws) 

        dims_ = OperatorStorage::GetRepeatedArgument<int>("dims");
        auto originalSize = dims_.size();
        CAFFE_ENFORCE_GT(originalSize, 0, "Parameter `dims` must be provided.");
        std::sort(dims_.begin(), dims_.end());
        dims_.erase(std::unique(dims_.begin(), dims_.end()), dims_.end());
        if (dims_.size() < originalSize) {
          LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
        }
        CAFFE_ENFORCE_GE(dims_.front(), 0, "Dimension ids must be non-negative.");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (!OperatorStorage::InputBlob(INPUT).template IsType<itensor>()) {
          return fallback_.Run(0);
        }

        const auto& X = Input(INPUT);
        auto* Y = Output(OUTPUT);
        if (&X != Y) {
          // Copy if not inplace
          ideep::direct_copy::compute(X, *Y);
        }
        if (dims_.empty()) {
          return true;
        }

        auto newDims = X.get_dims();
        CAFFE_ENFORCE_GE(
            newDims.size() + dims_.size(),
            dims_.back() + 1,
            "Input needs at least ",
            (1 + dims_.back() - dims_.size()),
            " dimensions given `dims`.");

        for (const auto dim : dims_) {
          newDims.insert(newDims.begin() + dim, 1);
        }

        Y->reshape(newDims);
        return true;
        */
    }
}


///-----------------------------------------
pub struct IDEEPSqueezeOp {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,

    dims:     Vec<i32>,
    fallback: <Self as FallbackOp>::Fallback,

}

impl FallbackOp for IDEEPSqueezeOp {
    type Fallback = IDEEPFallbackOp<SqueezeOp<CPUContext>, dyn SkipIndices<0>>;
}

input_tags!{
    IDEEPSqueezeOp {
        Input
    }
}

output_tags!{
    IDEEPSqueezeOp {
        Output
    }
}

impl IDEEPSqueezeOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            fallback_(operator_def, ws) 

        dims_ = OperatorStorage::GetRepeatedArgument<int>("dims");
        auto originalSize = dims_.size();
        CAFFE_ENFORCE_GT(originalSize, 0, "Parameter `dims` must be provided.");

        std::sort(dims_.begin(), dims_.end());
        dims_.erase(std::unique(dims_.begin(), dims_.end()), dims_.end());
        if (dims_.size() < originalSize) {
          LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
        }
        CAFFE_ENFORCE_GE(dims_.front(), 0, "Dimension ids must be non-negative.");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (!OperatorStorage::InputBlob(INPUT).template IsType<itensor>()) {
          return fallback_.Run(0);
        }

        const auto& X = Input(INPUT);
        auto* Y = Output(OUTPUT);

        CAFFE_ENFORCE_GT(
            X.ndims(),
            dims_.back(),
            "Input needs at least ",
            (dims_.back() + 1),
            " dimensions.");
        const auto& ideep_dims = X.get_dims();
        std::vector<int64_t> dims(ideep_dims.begin(), ideep_dims.end());
        const auto new_dims = SqueezeOp<IDEEPContext>::ComputeDims(dims, dims_);
        itensor::dims new_dims_ideep(new_dims.begin(), new_dims.end());
        if (&X != Y) {
          // Copy if not inplace
          ideep::direct_copy::compute(X, *Y);
        }

        Y->reshape(new_dims_ideep);
        return true;
        */
    }
}

register_ideep_operator!{ExpandDims, IDEEPExpandDimsOp}

register_ideep_operator!{Squeeze,    IDEEPSqueezeOp}
