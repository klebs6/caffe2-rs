crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Embedding.cpp]

pub fn embedding(
        weight:             &Tensor,
        indices:            &Tensor,
        padding_idx:        i64,
        scale_grad_by_freq: bool,
        sparse:             bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(weight.dim() == 2,  "'weight' must be 2-D");
      auto indices_arg = TensorArg(indices, "indices", 1);
      checkScalarTypes("embedding", indices_arg, {kLong, kInt});

      // TODO: use tensor.index() after improving perf
      if (indices.dim() == 1) {
        return weight.index_select(0, indices);
      }

      auto size = indices.sizes().vec();
      for (auto d : weight.sizes().slice(1)) {
        size.push_back(d);
      }

      return weight.index_select(0, indices.reshape(-1)).view(size);
        */
}

pub fn embedding_backward(
        grad:               &Tensor,
        indices:            &Tensor,
        num_weights:        i64,
        padding_idx:        i64,
        scale_grad_by_freq: bool,
        sparse:             bool) -> Tensor {
    
    todo!();
        /*
            if (sparse) {
        return embedding_sparse_backward(
            grad, indices, num_weights, padding_idx, scale_grad_by_freq);
      } else {
        return embedding_dense_backward(
            grad, indices, num_weights, padding_idx, scale_grad_by_freq);
      }
        */
}

pub fn embedding_sparse_backward(
        grad:               &Tensor,
        indices:            &Tensor,
        num_weights:        i64,
        padding_idx:        i64,
        scale_grad_by_freq: bool) -> Tensor {
    
    todo!();
        /*
            auto indices_arg = TensorArg(indices_, "indices", 2);
      checkScalarTypes("embedding_backward", indices_arg, {kLong, kInt});

      // TODO: implement scale_grad_by_freq
      if (scale_grad_by_freq) {
        AT_ERROR(
            "embedding_backward: scale_grad_by_freq not supported with sparse gradients");
      }

      Tensor indices = indices_;
      Tensor grad = grad_;
      if (padding_idx != -1) {
        TorchList<optional<Tensor>> c({indices != padding_idx});
        indices = indices.index(c);
        grad = grad.index(c);
      }

      i64 num_features = grad_.size(-1);
      auto weight_size = array<i64, 2>{{ num_weights, num_features }};
      auto dense_options = grad.options();

      // check if all our grad come from padding_idx
      if (grad.numel() == 0) {
        return _sparse_coo_tensor_unsafe(empty({1, 0}, indices_.options().dtype(kLong)),
                                             empty({0, num_features}, dense_options),
                                             weight_size);
      }

      auto index = indices.reshape({1, -1});
      auto values = grad.reshape({-1, num_features});
      return _sparse_coo_tensor_unsafe(index.to(kLong), values, weight_size);
        */
}

pub fn embedding_dense_backward_cpu(
        grad:               &Tensor,
        indices:            &Tensor,
        num_weights:        i64,
        padding_idx:        i64,
        scale_grad_by_freq: bool) -> Tensor {
    
    todo!();
        /*
            auto indices_arg = TensorArg(indices, "indices", 2);
      checkScalarTypes("embedding_backward", indices_arg, {kLong, kInt});

      auto grad_weight = zeros({num_weights, grad_.size(-1)}, grad_.options());
      auto indices_contig = indices.contiguous();
      i64 numel = indices.numel();
      auto grad = grad_.contiguous().view({numel, grad_.size(-1)});

      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_dense_backward_cpu", [&] () {
        auto indices_data = indices_contig.data_ptr<Index>();

        unique_ptr<Index[]> counts;
        if (scale_grad_by_freq) {
          counts.reset(new Index[num_weights]);
          for (const auto i : irange(numel)) {
            counts[indices_data[i]] = 0;
          }
          for (const auto i : irange(numel)) {
            counts[indices_data[i]]++;
          }
        }

        auto parallel_section = [&](Index start, Index end) {
          for (i64 i = 0; i < numel; i++) {
            if (indices_data[i] != padding_idx) {
              Index k = indices_data[i];
              if (k >= start && k < end) {
                double scale = 1.0;
                if (scale_grad_by_freq) {
                  scale /= counts[k];
                }
                grad_weight[k].add_(grad[i], scale);
              }
            }
          }
        };

        if (numel > 1000) {
          parallel_for(0, num_weights, 0, parallel_section);
        } else {
          parallel_section(0, num_weights);
        }
      });

      return grad_weight;
        */
}

pub fn embedding_renorm_cpu(
        self_:     &mut Tensor,
        indices:   &Tensor,
        max_norm:  f64,
        norm_type: f64) -> &mut Tensor {
    
    todo!();
        /*
            auto self_arg = TensorArg(self, "self", 1);
      auto indices_arg = TensorArg(indices, "indices", 2);
      checkDim("embedding_renorm_", self_arg, 2);
      checkScalarTypes("embedding_renorm_", indices_arg, {kLong, kInt});

      auto indices_contig = indices.contiguous();
      auto num_indices = indices.numel();

      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_renorm_cpu_", [&]() {
        auto data_ptr = indices_contig.data_ptr<Index>();
        auto sorted_indices = vector<Index>(data_ptr, data_ptr + num_indices);
        sort(sorted_indices.begin(), sorted_indices.end());

        // Note that we cannot use parallel_for here because we perform operations on
        // Tensor inside the loop. See github.com/pytorch/pytorch/issues/28370 for more details.
        for (auto i = 0; i < num_indices; i++) {
          if (i > 0 && sorted_indices[i] == sorted_indices[i - 1]) {
            continue;
          }
          auto row = self[sorted_indices[i]];
          auto norm = row.norm(norm_type).item<double>();
          if (norm > max_norm) {
            auto scale = max_norm / (norm + 1e-7);
            row *= scale;
          }
        }
      });

      return self;
        */
}
