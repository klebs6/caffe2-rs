crate::ix!();

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
