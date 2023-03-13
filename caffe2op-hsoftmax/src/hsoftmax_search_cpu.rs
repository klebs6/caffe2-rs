crate::ix!();

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
