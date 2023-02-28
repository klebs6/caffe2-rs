crate::ix!();

use crate::{
    OperatorStorage,
};

pub struct NGramFromCategoricalOp<F,T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
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

impl<F,T,Context> NGramFromCategoricalOp<F,T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            col_ids_(this->template GetRepeatedArgument<int>("col_ids")),
            categorical_limits_( this->template GetRepeatedArgument<int>("categorical_limits")),
            vals_(this->template GetRepeatedArgument<int>("vals")) 

        col_num_ = col_ids_.size();
        max_col_id_ = *std::max_element(col_ids_.begin(), col_ids_.end());
        CAFFE_ENFORCE_EQ(col_num_, categorical_limits_.size());
        int expected_vals_size = 0;
        for (auto& l : categorical_limits_) {
          CAFFE_ENFORCE_GT(l, 0);
          expected_vals_size += l;
        }
        CAFFE_ENFORCE_EQ(expected_vals_size, vals_.size());
        // compute ngram maps with small end
        for (auto& j : col_ids_) {
          CAFFE_ENFORCE_GE(j, 0);
          ngram_maps_.push_back(std::map<int, int>());
        }
        int base = 1;
        int idx = 0;
        for (int k = 0; k < col_num_; k++) {
          int l = categorical_limits_[k];
          for (int m = 0; m < l; m++) {
            int v = vals_[idx++];
            ngram_maps_[k][v] = m * base;
          }
          base *= l;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& floats = Input(0);
        auto N = floats.size(0);
        auto D = floats.size_from_dim(1);
        const F* floats_data = floats.template data<F>();

        auto* output = Output(0, {N}, at::dtype<T>());
        auto* output_data = output->template mutable_data<T>();
        math::Set<T, Context>(output->numel(), 0, output_data, &context_);

        CAFFE_ENFORCE_GT(D, max_col_id_);
        for (int i = 0; i < N; i++) {
          for (int k = 0; k < col_num_; k++) {
            int j = col_ids_[k];
            int v = round(floats_data[i * D + j]);
            // for out-of-vocabulary values, we always treat them the same as the
            // first value specified in vals; if we want to mimic the behavior as
            // sigrid NGram transform, just push front a random/impossible value at
            // each segments of vals
            output_data[i] += ngram_maps_[k].find(v) == ngram_maps_[k].end()
                ? 0
                : ngram_maps_[k][v];
          }
        }
        return true;
        */
    }
}
