crate::ix!();

/**
  | Given a sparse matrix, apply Lp regularization.
  | 
  | Currently only L1 and L2 are implemented.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SparseLpRegularizerOp<T,Context> {
    storage:     OperatorStorage,
    context:     Context,
    p:           f32,
    reg_lambda:  f32,
    phantom:     PhantomData<T>,
}

impl<T,Context> SparseLpRegularizerOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            p_(this->template GetSingleArgument<float>("p", 2.0)),
            reg_lambda_( this->template GetSingleArgument<float>("reg_lambda", 1e-5)) 

        CAFFE_ENFORCE(
            p_ == 1.0 || p_ == 2.0,
            "Sparse Lp regularizer only implemented for p=1 or p=2.");
        CAFFE_ENFORCE_GT(
            reg_lambda_,
            0.0,
            "Lambda for sparse Lp regularizer must be greater than 0.");
        CAFFE_ENFORCE_LT(
            reg_lambda_,
            1.0,
            "Lambda for sparse Lp regularizer must be less than 1.");
        */
    }
}
