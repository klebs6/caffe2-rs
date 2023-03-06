crate::ix!();

/**
  | It implements the LambdaRank as appeared
  | in Wu,
  | 
  | Qiang, et al. "Adapting boosting for
  | information retrieval measures."
  | Information Retrieval 13.3 (2010):
  | 254-270.
  | 
  | This method heuristically optimizes
  | the NDCG.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LambdaRankNdcgOp<T,Context> {
    storage:                  OperatorStorage,
    context:                  Context,
    use_ndcg_as_loss:         bool,
    use_idcg_normalization:   bool,
    use_exp_gain:             bool,
    gain:                     Tensor,
    discount:                 Tensor,
    rank_idx:                 Tensor,
    ideal_idx:                Tensor,
    lambda:                   Tensor,
    inv_log_i:                Tensor,
    phantom:                  PhantomData<T>,
}

num_inputs!{LambdaRankNdcg, 3}

num_outputs!{LambdaRankNdcg, 2}

input_tags!{
    LambdaRankNdcgOp {
        Pred,
        Rel,
        SessionLens
    }
}

output_tags!{
    LambdaRankNdcgOp {
        Loss,
        Dpred
    }
}

impl<T,Context> LambdaRankNdcgOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            use_ndcg_as_loss_(
                this->template GetSingleArgument<bool>("use_ndcg_as_loss", false)),
            use_idcg_normalization_(this->template GetSingleArgument<bool>(
                "use_idcg_normalization",
                true)),
            use_exp_gain_(
                this->template GetSingleArgument<bool>("use_exp_gain", true))
        */
    }
}
