crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct NGramFromCategoricalOp<F,T,Context> {
    storage:             OperatorStorage,
    context:             Context,
    col_ids:             Vec<i32>,
    categorical_limits:  Vec<i32>,
    vals:                Vec<i32>,
    ngram_maps:          Vec<HashMap<i32,i32>>,
    col_num:             i32,
    max_col_id:          i32,
    phantom:             PhantomData<T>,
    phantomF:            PhantomData<F>,
}

register_cpu_operator!{
    NGramFromCategorical,
    NGramFromCategoricalOp<f32, i64, CPUContext>
}

no_gradient!{NGramFromCategorical}

num_inputs!{NGramFromCategorical, 1}

num_outputs!{NGramFromCategorical, 1}

