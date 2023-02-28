crate::ix!();

use crate::{
    OperatorStorage,
    GradientMakerBase,
    OperatorDef,
    Tensor,
    CPUContext,
    TreeProto,
    NodeProto,
    PathProto
};

pub struct HSoftmaxOpBase<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
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

/**
  | Hierarchical softmax is an operator
  | which approximates the softmax operator
  | while giving significant training
  | speed gains and reasonably comparable
  | performance. In this operator, instead
  | of calculating the probabilities of
  | all the classes, we calculate the probability
  | of each step in the path from root to the
  | target word in the hierarchy.
  | 
  | The operator takes a 2-D tensor (Tensor)
  | containing a batch of layers, a set of
  | parameters represented by the weight
  | matrix and bias terms, and a 1-D tensor
  | (Tensor) holding labels, or the indices
  | of the target class. The hierarchy has
  | to be specified as an argument to the
  | operator.
  | 
  | The operator returns a 1-D tensor holding
  | the computed log probability of the
  | target class and a 2-D tensor of intermediate
  | outputs (from the weight matrix and
  | softmax from each step in the path from
  | root to target class) which will be used
  | by the gradient operator to compute
  | gradients for all samples in the batch.
  |
  */
pub struct HSoftmaxOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    base:    HSoftmaxOpBase<T, Context>,

    phantom: PhantomData<T>,
}

num_inputs!{HSoftmax, 4}

num_outputs!{HSoftmax, 2}

inputs!{HSoftmax, 
    0 => ("X",                   "Input data from previous layer"),
    1 => ("W",                   "2D blob containing 'stacked' fully connected weight matrices. Each node in the hierarchy contributes one FC weight matrix if it has children nodes. Dimension is N*D, D is input dimension of data (X), N is sum of all output dimensions, or total number of nodes (excl root)"),
    2 => ("b",                   "1D blob with N parameters"),
    3 => ("labels",              "int word_id of the target word")
}

outputs!{HSoftmax, 
    0 => ("Y",                   "1-D of log probability outputs, one per sample"),
    1 => ("intermediate_output", "Extra blob to store the intermediate FC and softmax outputs for each node in the hierarchical path of a word. The outputs from samples are stored in consecutive blocks in the forward pass and are used in reverse order in the backward gradientOp pass")
}

args!{HSoftmax, 
    0 => ("hierarchy",           "Serialized HierarchyProto string containing list of vocabulary words and their paths from root of hierarchy to the leaf")
}

impl HSoftmaxOp<f32, CPUContext> {

    #[inline] pub fn run_forward_single(
        &mut self, 
        x:                 *const f32,
        w:                 *const f32,
        b:                 *const f32,
        target:            i32,
        int_output:        *mut f32,
        bias_multiplier:   *const f32,
        dim_out:           i32,
        dim_in:            i32,
        int_output_offset: &mut i32) -> f32 
    {
        todo!();
        /*
            // W * x
      float* fc_output_data = int_output + int_output_offset;

      math::Gemm<float, CPUContext>(CblasNoTrans, CblasTrans, 1, dim_out, dim_in, 1,
        X, W, 0, fc_output_data, &context_);
      math::Gemv<float, CPUContext>(CblasNoTrans, dim_out, 1, 1,
        b, bias_multiplier, 1, fc_output_data, &context_);

      int_output_offset += dim_out;

      //Softmax
      float* softmax_output_data = int_output + int_output_offset;

      if (!scale_.has_value()) {
        scale_ = caffe2::empty({1}, at::dtype<float>().device(CPU));
      }

      if (!sum_multiplier_.has_value()) {
        sum_multiplier_ = caffe2::empty({dim_out}, at::dtype<float>().device(CPU));
        math::Set<float, CPUContext>(dim_out, 1.f,
          sum_multiplier_->mutable_data<float>(), &context_);
      } else if (sum_multiplier_->numel() != dim_out) {
        sum_multiplier_->Resize(dim_out);
        math::Set<float, CPUContext>(dim_out, 1.f,
          sum_multiplier_->mutable_data<float>(), &context_);
      }
      math::RowwiseMax<float, CPUContext>(1, dim_out, fc_output_data,
        scale_->mutable_data<float>(), &context_);

      // Put the intermediate result X - max(X) into Y
      context_.template CopyFromCPU<float>(
          dim_out, fc_output_data, softmax_output_data);
      // Subtract the scale
      math::Gemv<float, CPUContext>(CblasNoTrans, dim_out, 1, -1,
        sum_multiplier_->data<float>(), scale_->data<float>(), 1, softmax_output_data,
        &context_);

      // Exponentiation
      math::Exp<float, CPUContext>(dim_out, softmax_output_data,
        softmax_output_data, &context_);
      math::Gemv<float, CPUContext>(CblasNoTrans, 1, dim_out, 1,
        softmax_output_data, sum_multiplier_->data<float>(), 0,
        scale_->mutable_data<float>(), &context_);

      // Do division
      const float scale = *(scale_->data<float>());
      for (int j = 0; j < dim_out; ++j) {
        softmax_output_data[j] /= scale;
      }

      int_output_offset += dim_out;

      if (target < 0) {
        return -1;
      }
      //Return cross entropy loss
      return -log(std::max(softmax_output_data[target], kLOG_THRESHOLD()));
        */
    }

    /// Implementation for the CPU context.
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      const auto& W = Input(1);
      const auto& b = Input(2);
      auto& label = Input(3);

      // Batch size
      int M = X.dim() > 1 ? X.dim32(0) : 1;
      // Input feature dimension
      size_t K = X.numel() / M;
      CAFFE_ENFORCE_GE(W.dim(), 2); // N*K
      CAFFE_ENFORCE_EQ(b.dim(), 1); // N
      CAFFE_ENFORCE_EQ(K, W.numel() / (W.dim32(0)));
      // Sum of output dimensions of all hierarchy nodes
      int N = W.dim32(0);
      CAFFE_ENFORCE_EQ(N, b.dim32(0));
      auto* Y = Output(0, {M}, at::dtype<float>());
      auto* Ydata = Y->template mutable_data<float>();
      math::Set<float, CPUContext>(M, 0.f, Ydata, &context_);
      const auto* labeldata = label.data<int>();

      auto hierarchy = getHierarchyForLabels(M, labeldata, hierarchy_all_map_);
      int int_output_size = getIntermediateOutputSize(labeldata, M, hierarchy);
      auto* intermediate_output = Output(1, {int_output_size}, at::dtype<float>());
      float* int_output_data = intermediate_output->template mutable_data<float>();
      int int_output_offset = 0;

      if (!bias_multiplier_.has_value()) {
        bias_multiplier_ = caffe2::empty({M}, at::dtype<float>().device(CPU));
        math::Set<float, CPUContext>(M, static_cast<float>(1),
            bias_multiplier_->mutable_data<float>(), &context_);
      } else if (bias_multiplier_->numel() != M) {
        bias_multiplier_->Resize(M);
        math::Set<float, CPUContext>(M, static_cast<float>(1),
            bias_multiplier_->mutable_data<float>(), &context_);
      }

      for (int sample = 0; sample < M; ++sample) {
        int word_id = labeldata[sample];
        const PathProto& path = hierarchy[word_id];
        for (const PathNodeProto& node : path.path_nodes()) {
          //Offset of node's weight matrix in W
          int w_offset = node.index();
          //Number of output dimensions in node's weight matrix
          int w_length = node.length();
          int target = node.target();
          //Adding log probabilities
          Ydata[sample] += RunForwardSingle(X.data<float>() + sample*K,
            W.data<float>() + w_offset*K, b.data<float>() + w_offset, target,
            int_output_data, bias_multiplier_->data<float>()+sample, w_length, K,
            int_output_offset);
        }
      }
      return true;
        */
    }
}

///------------------------------------------
pub struct HSoftmaxGradientOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    base: HSoftmaxOpBase<T, Context>,

    phantom: PhantomData<T>,
}

num_inputs!{HSoftmaxGradient, 6}

num_outputs!{HSoftmaxGradient, 4}

impl HSoftmaxGradientOp<f32, CPUContext> {

    #[inline] pub fn run_backward_single(
        &mut self, 
        x:                  *const f32,
        dY:                 *const f32,
        w:                  *const f32,
        target:             i32,
        int_output:         *const f32,
        dX:                 *mut f32,
        dW:                 *mut f32,
        db:                 *mut f32,
        dint_output:        *mut f32,
        dim_in:             i32,
        dim_out:            i32,
        int_output_offset:  &mut i32)  
    {
        
        todo!();
        /*
            //Cross entropy
      // dX_entropy is the dX for the cross entropy layer
      float* dX_entropy = dint_output + int_output_offset - dim_out;
      // X_entropy is the X for the cross entropy layer and Y for the softmax layer
      const float* X_entropy = int_output + int_output_offset - dim_out;

      math::Set<float, CPUContext>(dim_out, 0.f, dX_entropy, &context_);
      dX_entropy[target] = - (*dY) / std::max(X_entropy[target], kLOG_THRESHOLD());

      int_output_offset -= dim_out;

      //Softmax
      if (!scale_.has_value()) {
        scale_ = caffe2::empty({1}, at::dtype<float>().device(CPU));
      }
      float* scaledata = scale_->mutable_data<float>();

      if (!sum_multiplier_.has_value()) {
        sum_multiplier_ = caffe2::empty({dim_out}, at::dtype<float>().device(CPU));
        math::Set<float, CPUContext>(dim_out, 1.f,
          sum_multiplier_->mutable_data<float>(), &context_);
      } else if (sum_multiplier_->numel() != dim_out) {
        sum_multiplier_->Resize(dim_out);
        math::Set<float, CPUContext>(dim_out, 1.f,
          sum_multiplier_->mutable_data<float>(), &context_);
      }

      float* dX_softmax = dint_output + int_output_offset - dim_out;
      context_.CopyFromCPU<float>(dim_out, dX_entropy, dX_softmax);

      math::Dot<float, CPUContext>(dim_out, X_entropy, dX_entropy, scaledata,
        &context_);
      math::Gemv<float, CPUContext>(CblasTrans, 1, dim_out, -1,
        sum_multiplier_->data<float>(), scaledata , 1, dX_softmax, &context_);
      math::Mul<float, CPUContext>(dim_out, dX_softmax, X_entropy, dX_softmax,
        &context_);

      int_output_offset -= dim_out;

      //FC
      if (!bias_multiplier_.has_value()) {
        // If the helper bias multiplier has not been created, reshape and fill
        // it with 1
        bias_multiplier_ = caffe2::empty({1}, at::dtype<float>().device(CPU));
        math::Set<float, CPUContext>(1, static_cast<float>(1),
            bias_multiplier_->template mutable_data<float>(), &context_);
      }

      // Compute dW and add incrementally
      // dW = dW + dX_softmax'*X
      math::Gemm<float, CPUContext>(CblasTrans, CblasNoTrans, dim_out, dim_in, 1, 1,
        dX_softmax, X, 1, dW, &context_);

      // Compute dB and add incrementally
      // db = db + dX_softmax*bias_multiplier_
      math::Gemv<float, CPUContext>(CblasTrans, 1, dim_out, 1, dX_softmax,
        bias_multiplier_->template data<float>(), 1, db, &context_);

      // Compute dX and add incrementally
      // dX = dX + W'dX_softmax
      math::Gemv<float, CPUContext>(CblasTrans, dim_out, dim_in,
        1, W, dX_softmax, 1, dX, &context_);
        */
    }

    /// Implementation for the CPU context.
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      const auto& W = Input(1);
      const auto& b = Input(2);
      auto& label = Input(3);
      auto& intermediate_output = Input(4);
      auto& dY = Input(5);

      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      auto* dW = Output(1, W.sizes(), at::dtype<float>());
      auto* db = Output(2, b.sizes(), at::dtype<float>());
      auto* dX_intermediate_output =
          Output(3, intermediate_output.sizes(), at::dtype<float>());

      float* dX_data = dX->template mutable_data<float>();
      float* dW_data = dW->template mutable_data<float>();
      float* db_data = db->template mutable_data<float>();
      float* dOutput_data = dX_intermediate_output->template mutable_data<float>();

      math::Set<float, CPUContext>(X.numel(), 0.f, dX_data, &context_);
      math::Set<float, CPUContext>(W.numel(), 0.f, dW_data, &context_);
      math::Set<float, CPUContext>(b.numel(), 0.f, db_data, &context_);
      math::Set<float, CPUContext>(
          intermediate_output.numel(), 0.f, dOutput_data, &context_);

      // Batch size
      int M = X.dim() > 1 ? X.dim32(0) : 1;
      // Input feature dimension
      int K = X.numel() / M;
      const auto* labeldata = label.data<int>();

      auto hierarchy = getHierarchyForLabels(M, labeldata, hierarchy_all_map_);
      int output_offset = getIntermediateOutputSize(labeldata, M, hierarchy);

      //Traverse backward to access intermediate_output generated by HSoftmaxOp
      // sequentially in reverse order
      for (int sample = M-1; sample >= 0; sample--) {
        int word_id = labeldata[sample];
        PathProto path = hierarchy[word_id];
        for (auto node = path.path_nodes().rbegin();
          node != path.path_nodes().rend(); node++) {
          int w_offset = node->index();
          int w_length = node->length();
          int target = node->target();
          RunBackwardSingle(X.data<float>() + sample*K, dY.data<float>() + sample,
            W.data<float>() + w_offset*K, target, intermediate_output.data<float>(),
            dX_data + sample*K, dW_data + w_offset*K, db_data + w_offset,
            dOutput_data, K, w_length, output_offset);
        }
      }
      return true;
        */
    }
}

/**
  | HSoftmaxSearch is an operator to generate
  | the most possible paths given a well-trained
  | model and input vector. Greedy algorithm
  | is used for pruning the search tree.
  |
  */
pub struct HSoftmaxSearchOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    base:    HSoftmaxOp<T, Context>,

    top_n:   i32,
    beam:    f32,
    tree:    TreeProto,

    phantom: PhantomData<T>,
}

num_inputs!{HSoftmaxSearch, 3}

num_outputs!{HSoftmaxSearch, 2}

inputs!{HSoftmaxSearch, 
    0 => ("X", "Input data from previous layer"),
    1 => ("W", "The matrix trained from Softmax Ops"),
    2 => ("b", "The bias trained from Softmax Ops")
}

outputs!{HSoftmaxSearch, 
    0 => ("Y_names", "The name of selected nodes and leafs. For nodes, it will be the name defined in the tree. For leafs, it will be the index of the word in the tree."),
    1 => ("Y_scores", "The corresponding scores of Y_names")
}

args!{HSoftmaxSearch, 
    0 => ("tree", "Serialized TreeProto string containing a tree including all intermidate nodes and leafs. All nodes must have names for correct outputs"),
    1 => ("beam", "beam used for pruning tree. The pruning algorithm is that only children, whose score is smaller than parent's score puls beam, will be propagated. "),
    2 => ("topN", "Number of nodes in outputs")
}

should_not_do_gradient!{HSoftmaxSearch}

impl<T, Context> HSoftmaxSearchOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : HSoftmaxOp<T, Context>(std::forward<Args>(args)...),
            top_n_(this->template GetSingleArgument<int>("topN", 5)),
            beam_(this->template GetSingleArgument<float>("beam", 0.01f)) 

        CAFFE_ENFORCE(tree_.ParseFromString(
            this->template GetSingleArgument<string>("tree", "")));
        */
    }
}

impl HSoftmaxSearchOp<f32, CPUContext> {

    /// Implementation for the CPU context.
    #[inline] pub fn pruning(
        &mut self, 
        x:               *const f32,
        sample:          i32,
        k:               i32,
        w:               *const f32,
        b:               *const f32,
        src_node:        &NodeProto,
        dst_node:        &mut NodeProto,
        parent_score:    f32,
        beam:            f32) -> bool 
    {
        todo!();
        /*
            int w_length = src_node.children_size() + src_node.word_ids_size();
      Tensor intermediate_data{CPU};
      intermediate_data.Resize(2 * w_length);
      float* int_output_data = intermediate_data.template mutable_data<float>();
      int int_output_offset = 0;
      int w_offset = src_node.offset();

      RunForwardSingle(
          X + K * sample,
          W + w_offset * K,
          b + w_offset,
          -1,
          int_output_data,
          bias_multiplier_->template data<float>() + sample,
          w_length,
          K,
          int_output_offset);

      float* softmax_output_data = int_output_data + w_length;
      // real probabilities
      for (int i = 0; i < w_length; i++) {
        softmax_output_data[i] =
            -log(std::max(softmax_output_data[i], kLOG_THRESHOLD())) + parent_score;
      }
      for (int i = 0; i < src_node.children_size(); i++) {
        if (softmax_output_data[i] < parent_score + beam) {
          dst_node.add_children();
          int idx = dst_node.children_size() - 1;
          CAFFE_ENFORCE(
              src_node.children(i).has_offset(),
              "HSM Search require the field offset in NodeProte");
          dst_node.mutable_children(idx)->set_offset(src_node.children(i).offset());
          CAFFE_ENFORCE(
              src_node.children(i).has_name(),
              "HSM Search require the field name in NodeProte");
          dst_node.mutable_children(idx)->set_name(src_node.children(i).name());
          dst_node.add_scores(softmax_output_data[i]);
          pruning(
              X,
              sample,
              K,
              W,
              b,
              src_node.children(i),
              *dst_node.mutable_children(idx),
              softmax_output_data[i],
              beam);
        }
      }

      for (int i = src_node.children_size(); i < w_length; i++) {
        if (softmax_output_data[i] < parent_score + beam) {
          dst_node.add_word_ids(src_node.word_ids(i - src_node.children_size()));
          dst_node.add_scores(softmax_output_data[i]);
        }
      }

      return true;
        */
    }
    
    #[inline] pub fn extract_nodes(
        &mut self, 
        node: &NodeProto,
        info: &mut Vec<(String,f32)>) -> bool {

        todo!();
        /*
            int i = 0;

      for (const auto& n : node.children()) {
        info.emplace_back(std::make_pair(n.name(), node.scores(i++)));
      }
      for (const int n : node.word_ids()) {
        info.emplace_back(std::make_pair(c10::to_string(n), node.scores(i++)));
      }

      for (const auto& n : node.children()) {
        extractNodes(n, info);
      }
      return true;
        */
    }

    /// Implementation for the CPU context.
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      const auto& W = Input(1);
      const auto& b = Input(2);

      // Batch size
      int M = X.dim() > 1 ? X.dim32(0) : 1;
      // Input feature dimension
      int K = X.numel() / M;
      CAFFE_ENFORCE(W.dim() == 2, "Weight must be a matrix."); // N*K
      CAFFE_ENFORCE(b.dim() == 1, "Bias must be a vector."); // N
      CAFFE_ENFORCE(K == W.numel() / (W.dim32(0)), "feature dimension mismatch.");
      // Sum of output dimensions of all hierarchy nodes
      int N = W.dim32(0);
      CAFFE_ENFORCE(N == b.dim32(0), "mismatch between Weight and Bias.");
      auto* Y_names = Output(0, {M, top_n_}, at::dtype<string>());
      auto* Y_scores = Output(1, {M, top_n_}, at::dtype<float>());

      if (!bias_multiplier_.has_value()) {
        bias_multiplier_ = caffe2::empty({M}, at::dtype<float>().device(CPU));
        math::Set<float, CPUContext>(M, static_cast<float>(1),
            bias_multiplier_->mutable_data<float>(), &context_);
      } else if (bias_multiplier_->numel() != M) {
        bias_multiplier_->Resize(M);
        math::Set<float, CPUContext>(M, static_cast<float>(1),
            bias_multiplier_->mutable_data<float>(), &context_);
      }

      for (int sample = 0; sample < M; ++sample) {
        CAFFE_ENFORCE(
            tree_.root_node().has_offset(),
            "HSM Search require the field offset in NodeProte");
        CAFFE_ENFORCE(
            tree_.root_node().has_name(),
            "HSM Search require the field name in NodeProte");

        NodeProto dst_node;
        dst_node.set_offset(tree_.root_node().offset());
        dst_node.set_name(tree_.root_node().name());

        pruning(
            X.data<float>(),
            sample,
            K,
            W.data<float>(),
            b.data<float>(),
            tree_.root_node(),
            dst_node,
            0,
            beam_);

        std::vector<std::pair<string, float>> info;
        extractNodes(dst_node, info);
        // saving the results for each sample.
        std::partial_sort(
            info.begin(),
            info.begin() + (top_n_ < info.size() ? top_n_ : info.size() - 1),
            info.end(),
            [&](std::pair<string, float> a, std::pair<string, float> b) {
              return a.second < b.second;
            });
        auto* y_name_data =
            Y_names->template mutable_data<string>() + sample * top_n_;
        auto* y_score_data =
            Y_scores->template mutable_data<float>() + sample * top_n_;
        for (int i = 0; i < top_n_; i++) {
          if (i < info.size()) {
            y_name_data[i] = info[i].first;
            y_score_data[i] = info[i].second;
          } else {
            y_score_data[i] = 0;
          }
        }
      }

      return true;
        */
    }
}

/**
  | HuffmanTreeHierarchy is an operator
  | to generate huffman tree hierarchy
  | given the input labels. It returns the
  | tree as serialized HierarchyProto
  |
  */
pub struct HuffmanTreeHierarchyOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:     OperatorStorage,
    context:     Context,

    num_classes: i32,

    phantom:     PhantomData<T>,
}

num_inputs!{HuffmanTreeHierarchy, 1}

num_outputs!{HuffmanTreeHierarchy, 1}

inputs!{HuffmanTreeHierarchy, 
    0 => ("Labels", "The labels vector")
}

outputs!{HuffmanTreeHierarchy, 
    0 => ("Hierarch", "Huffman coding hierarchy of the labels")
}

args!{HuffmanTreeHierarchy, 
    0 => ("num_classes", "The number of classes used to build the hierarchy.")
}

should_not_do_gradient!{HuffmanTreeHierarchyOp}

/// Internal huffman tree data.
mod huffman_tree_hierarchy {

    pub struct Node<T> {
        label:          T,
        count:          i32,
        left_ch_index:  i32,
        right_ch_index: i32,
    }

    impl<T> Node<T> {
        fn new(l: T, count: i32) -> Self {
            Self {
                label:          l,
                count:          count,
                left_ch_index:  -1,
                right_ch_index: -1,
            }
        }
    }

    pub struct NodeComparator;

    impl NodeComparator {

        #[inline] pub fn invoke<T>(
            &mut self, 
            node_a: &Node<T>,
            node_b: &Node<T>) -> bool 
        {
            todo!();
            /*
               return node_a.count > node_b.count;
               */
        }
    }
}

impl<T, Context> HuffmanTreeHierarchyOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            num_classes_(this->template GetSingleArgument<int>("num_classes", -1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& Y = Input(0);

      CAFFE_ENFORCE_EQ(Y.dim(), 1, "Input labels must be a vector.");
      const auto y_data = Y.template data<T>();
      auto treeOutput = Output(0, {1}, at::dtype<string>());
      std::vector<int> labelCounts;
      labelCounts.resize(num_classes_, 0);
      for (int i = 0; i < Y.dim32(0); ++i) {
        // Labels are in range [0, num_classes]
        const int label_index = y_data[i];
        CAFFE_ENFORCE_LT(
            label_index,
            num_classes_,
            "Found an input label ",
            label_index,
            " not in range [",
            0,
            ",",
            num_classes_,
            "]");
        labelCounts[label_index]++;
      }

      std::priority_queue<Node, std::vector<Node>, NodeComparator> nodes;
      std::vector<Node> huffmanTree;
      std::vector<int> labelIndices;
      labelIndices.resize(num_classes_);

      int current_node_index = 0;
      for (int i = 0; i < num_classes_; ++i) {
        Node node(i, labelCounts[i]);
        nodes.push(node);
      }

      // Extract node with minimum count and insert it in the tree array.
      auto get_next_node = [&nodes, &huffmanTree, &labelIndices]() {
        auto node = nodes.top();
        int node_index = huffmanTree.size();
        if (node.label != -1) {
          labelIndices[node.label] = node_index;
        }
        nodes.pop();
        huffmanTree.push_back(node);
        return std::pair<int, Node>(node_index, node);
      };

      // Merge two nodes and insert the results in the queue.
      auto merge_nodes = [&nodes](
          const std::pair<int, Node>& node_l, const std::pair<int, Node>& node_r) {
        Node node(-1, node_l.second.count + node_r.second.count);
        node.left_ch_index = node_l.first;
        node.right_ch_index = node_r.first;
        nodes.push(node);
      };

      // Main loop for buttom up huffman tree construction.
      while (!nodes.empty()) {
        auto lNode = get_next_node();
        if (!nodes.empty()) {
          auto rNode = get_next_node();
          merge_nodes(lNode, rNode);
        }
      }

      auto is_leaf_node = [&huffmanTree](const int node_index) {
        return huffmanTree[node_index].left_ch_index == -1 &&
            huffmanTree[node_index].right_ch_index == -1;
      };

      auto get_node_label = [&huffmanTree](const int node_index) {
        return huffmanTree[node_index].label;
      };

      // Build huffman tree.
      int current_offset = 0;
      std::function<void(int, NodeProto*)> build_tree = [&](
          const int node_index, NodeProto* node) {
        if (is_leaf_node(node_index) || node_index == -1) {
          return;
        }
        const int left_ch_index = huffmanTree[node_index].left_ch_index;
        const int right_ch_index = huffmanTree[node_index].right_ch_index;
        if (left_ch_index != -1) {
          if (is_leaf_node(left_ch_index)) {
            node->add_word_ids(get_node_label(left_ch_index));
          } else {
            auto* ch_node = node->add_children();
            ch_node->set_offset(current_offset);
            current_offset += 2;
            build_tree(left_ch_index, ch_node);
          }
        }
        if (right_ch_index != -1) {
          if (is_leaf_node(right_ch_index)) {
            node->add_word_ids(get_node_label(right_ch_index));
            current_offset++;
          } else {
            auto* ch_node = node->add_children();
            ch_node->set_offset(current_offset);
            current_offset += 2;
            build_tree(right_ch_index, ch_node);
          }
        }
      };

      // The last element inserted in the tree is the root.
      const int rootNodeIndex = huffmanTree.size() - 1;
      NodeProto rootNode;
      rootNode.set_offset(current_offset);
      current_offset += 2;
      build_tree(rootNodeIndex, &rootNode);
      TreeProto treeProto;
      *treeProto.mutable_root_node() = rootNode;

      treeProto.SerializeToString(treeOutput->template mutable_data<string>());
      return true;
        */
    }
}

register_cpu_operator!{HSoftmax,                HSoftmaxOp<f32, CPUContext>}
register_cpu_operator!{HSoftmaxGradient,        HSoftmaxGradientOp<f32, CPUContext>}
register_cpu_operator!{HSoftmaxSearch,          HSoftmaxSearchOp<f32, CPUContext>}
register_cpu_operator!{HuffmanTreeHierarchy,    HuffmanTreeHierarchyOp<int64_t, CPUContext>}

pub struct GetHSoftmaxGradient;

impl GetGradientDefs for GetHSoftmaxGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
                "HSoftmaxGradient", "",
                //X, W, b, label, intermediate output, dY
                vector<string>{I(0), I(1), I(2), I(3), O(1), GO(0)},
                //dX, dW, db, dintermediate_output
                vector<string>{GI(0), GI(1), GI(2), GO(1)});
        */
    }
}

register_gradient!{HSoftmax, GetHSoftmaxGradient}
