crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AccumulateInputGradientOp<Context> {
    storage: OperatorStorage,
    context: Context,
    offset:  i32,
}

impl<Context> AccumulateInputGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            offset_(this->template GetSingleArgument<int>("offset", -1)) 

        CAFFE_ENFORCE(offset_ >= 0, "Offset not set");
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& t0 = this->template Input<Tensor>(0, CPU);
            const auto t = t0.template data<int32_t>()[0];
            auto& og = Input(1);
            auto* g = Output(0);

            T* g_data = g->template mutable_data<T>();
            const auto timestep_size = g->numel() / g->size(0);

            CAFFE_ENFORCE(
                (t + offset_) * timestep_size + timestep_size <= g->numel(),
                "Accumulation destination address over bounds");
            CAFFE_ENFORCE(
                t * timestep_size + timestep_size <= og.numel(),
                "Accumulation source address out of bounds");

            math::Add<T, Context>(
                timestep_size,
                og.template data<T>() + t * timestep_size,
                g_data + (t + offset_) * timestep_size,
                g_data + (t + offset_) * timestep_size,
                &context_);
            return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float>>::call(this, Input(1));
        */
    }
}
