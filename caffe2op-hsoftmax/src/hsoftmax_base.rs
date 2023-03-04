crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct HSoftmaxOpBase<T, Context> {

    storage:           OperatorStorage,
    context:           Context,

    hierarchy_all_map: HashMap<i32,PathProto>,
    scale:             Option<Tensor>,
    sum_multiplier:    Option<Tensor>,
    bias_multiplier:   Option<Tensor>,

    phantom:           PhantomData<T>,
}

impl<T, Context> HSoftmaxOpBase<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        HierarchyProto hierarchy;
        CAFFE_ENFORCE(hierarchy.ParseFromString(
            this->template GetSingleArgument<string>("hierarchy", "")));
        for (const auto& path : hierarchy.paths()) {
          hierarchy_all_map_.emplace(path.word_id(), path);
        }
        */
    }
    
    #[inline] pub fn klog_threshold() -> T {
        
        todo!();
        /*
            return 1e-20f;
        */
    }
    
    #[inline] pub fn get_hierarchy_for_labels(
        m: i32,
        labels: *const i32,
        hierarchy_all_map: &HashMap<i32,PathProto>) -> HashMap<i32,PathProto> 
    {
        todo!();
        /*
            std::unordered_map<int, PathProto> hierarchy_map;
        std::set<int> label_set = std::set<int>(labels, labels + M);
        for (const auto& label : label_set) {
          auto search = hierarchy_all_map.find(label);
          CAFFE_ENFORCE(search != hierarchy_all_map.end(), "incorrect label.");
          hierarchy_map.emplace(search->first, search->second);
        }
        return hierarchy_map;
        */
    }
    
    #[inline] pub fn get_intermediate_output_size(
        &self, 
        labels: *const i32,
        m: i32,
        hierarchy: &mut HashMap<i32,PathProto>) -> i32 
    {
        todo!();
        /*
            int size = 0;
        for (int label = 0; label < M; ++label) {
          int word_id = labels[label];
          const auto& path = hierarchy[word_id];
          size += std::accumulate(
              path.path_nodes().begin(),
              path.path_nodes().end(),
              0,
              // Output of FC + Output of Softmax
              [](int sz, PathNodeProto node) { return sz + 2 * node.length(); });
        }
        return size;
        */
    }
}
