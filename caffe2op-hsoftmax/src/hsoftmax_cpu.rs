crate::ix!();

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
