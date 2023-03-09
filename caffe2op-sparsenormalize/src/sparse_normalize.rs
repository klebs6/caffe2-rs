crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SparseNormalizeOp<T,Context> {
    storage:       OperatorStorage,
    context:       Context,
    use_max_norm:  bool,
    norm:          f32,
    phantom:       PhantomData<T>,
}

impl<T,Context> SparseNormalizeOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            use_max_norm_( this->template GetSingleArgument<bool>("use_max_norm", true)),
            norm_(this->template GetSingleArgument<float>("norm", 1.0)) 

        CAFFE_ENFORCE_GE(norm_, 0, "norm should be bigger than 0");
        */
    }
}
