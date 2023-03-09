crate::ix!();

declare_export_caffe2_op_to_c10!{SparseToDenseMask}

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SparseToDenseMaskBase<Context> {
    storage:         OperatorStorage,
    context:         Context,
    sparse:          HashMap<i64,i32>,
    dense:           Vec<i32>,
    features_count:  usize,
}

impl<Context> SparseToDenseMaskBase<Context> {

    const kMaxDenseSize: i64 = 1024 * 128;

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        std::vector<int64_t> mask =
            this->template GetRepeatedArgument<int64_t>("mask");
        featuresCount_ = mask.size();

        CAFFE_ENFORCE(!mask.empty(), "mask can't be empty");
        auto biggest = *std::max_element(mask.begin(), mask.end());
        dense_.assign(std::min(kMaxDenseSize, biggest + 1), -1);
        for (int i = 0; i < mask.size(); i++) {
          int64_t id = mask[i];
          CAFFE_ENFORCE_GE(id, 0, "Only positive IDs are allowed.");
          if (id >= kMaxDenseSize) {
            CAFFE_ENFORCE(sparse_.count(id) == 0, "Duplicated id: ", id);
            sparse_[id] = i;
          } else {
            CAFFE_ENFORCE(dense_[id] == -1, "Duplicated id: ", id);
            dense_[id] = i;
          }
        }
        */
    }
    
    #[inline] pub fn get_feature_idx(&self, id: i64) -> i32 {
        
        todo!();
        /*
            if (id >= kMaxDenseSize) {
          const auto& iter = sparse_.find(id);
          if (iter == sparse_.end()) {
            return -1;
          } else {
            return iter->second;
          }
        } else {
          return (id >= dense_.size()) ? -1 : dense_[id];
        }
        */
    }
}
