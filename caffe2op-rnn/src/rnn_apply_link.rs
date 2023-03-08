crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RNNApplyLinkOp<Context> {
    storage: OperatorStorage,
    context: Context,
    offset:  i32,
    window:  i32,
}

impl<Context> RNNApplyLinkOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            offset_(this->template GetSingleArgument<int>("offset", -1)),
            window_(this->template GetSingleArgument<int>("window", -1)) 

        CAFFE_ENFORCE(offset_ >= 0, "offset not set");
        CAFFE_ENFORCE(window_ >= 0, "window not set");
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            // Both internal and external appear as both input and output to enforce
            // correct dependency computation.
            const auto& t0 = this->template Input<Tensor>(0, CPU);
            const auto t = t0.template data<int32_t>()[0];
            auto& external = Input(1);

            auto* internal_out = Output(0);
            auto* external_out = Output(1);

            CAFFE_ENFORCE_GT(external.numel(), 0);
            const int64_t externalTimestepSize = external.numel() / external.size(0);
            auto* externalData = external_out->template mutable_data<T>() +
                (t + offset_) * externalTimestepSize;
            auto internalDims = external_out->sizes().vec();
            internalDims[0] = window_;

            internal_out->Resize(internalDims);
            internal_out->ShareExternalPointer(
                externalData, externalTimestepSize * window_);
            return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<float>();
        */
    }
}
