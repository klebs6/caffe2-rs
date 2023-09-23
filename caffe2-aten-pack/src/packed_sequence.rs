crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/PackedSequence.cpp]

pub fn check_long_tensor(tensor: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(tensor.dim() == 1 && tensor.device().type() == kCPU && tensor.scalar_type() == kLong,
               "'lengths' argument should be a 1D CPU int64 tensor, but got ",
                tensor.dim(), "D ", tensor.device().str(), " ", tensor.scalar_type(), " tensor");
        */
}

/**
  | This method returns `(data, batch_sizes)`,
  | which are then passed into a `PackedSequence`
  | constructor.
  |
  | `data` can be on arbitrary device and of
  | arbitrary dtype, but `batch_sizes` must be
  | a CPU int64 tensor.
  |
  | See NOTE [ device and dtype of a PackedSequence
  | ]
  */
pub fn pack_padded_sequence(
        input:       &Tensor,
        lengths:     &Tensor,
        batch_first: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            auto input = batch_first ? _input.transpose(0, 1) : _input;
      auto lengths_t = _lengths.contiguous();
      checkLongTensor(lengths_t);

      i64 batch_size = input.size(1);
      i64 * lengths = lengths_t.data_ptr<i64>();
      TORCH_CHECK(input.numel() > 0, "Cannot pack empty tensors.");
      TORCH_CHECK(lengths_t.size(0) == batch_size,
               "Expected `len(lengths)` to be equal to batch_size, but got ", lengths_t.size(0),
               " (batch_size=", batch_size, ")");
      TORCH_CHECK(lengths[batch_size - 1] > 0,
               "Length of all samples has to be greater than 0, but found an element "
               "in 'lengths' that is <= 0");
      for (const auto i : irange(batch_size - 1)) {
        if (lengths[batch_size - 1 - i] > lengths[batch_size - 2 - i]) {
          // NB: enforce_sorted is implemented at a Python level, but the sortedness
          // check lives here. If enforce_sorted=False then this error should never
          // get called.
          AT_ERROR("`lengths` array must be sorted in decreasing order when "
                   "`enforce_sorted` is True. You can pass `enforce_sorted=False` "
                   "to pack_padded_sequence and/or pack_sequence to sidestep this "
                   "requirement if you do not need ONNX exportability.");
        }
      }

      vector<Tensor> steps;
      steps.reserve(batch_size);
      Tensor batch_sizes_t = empty(lengths[0], _lengths.options());
      i64 * batch_sizes = batch_sizes_t.data_ptr<i64>();

      vector<i64> step_shape; // == [-1, *input.shape[2:]]
      {
        auto input_sizes = input.sizes();
        step_shape.reserve(input_sizes.size());
        auto s_input_sizes = input_sizes.slice(2);
        step_shape.push_back(-1);
        step_shape.insert(step_shape.end(), s_input_sizes.begin(), s_input_sizes.end());
      }

      // To understand what's going on in this loop imagine that the input is a padded 2D
      // array that looks like this (x = valid entry, . = padding)
      //
      //  1 1 1 1 1
      //  2 2 2 . .
      //  2 2 2 . .
      //  4 . . . .
      //  4 . . . .
      //
      // Where the vertical dimension corresponds to time, and horizontal dim to batch.
      // In this example, the lengths array will be equal to [5, 3, 3, 1, 1], and we will
      // iterate over them in reverse order (from the rightmost column to the left).
      // We want to avoid eager slicing of the input at every time step, and wait for
      // the moments where the length increases. In this example, that will happen at the
      // first, second and fourth steps. Then, we slice out the whole block of the input
      // that corresponds to this length, and hasn't been sliced yet (the steps at which each
      // element is sliced are annotated in the array above).  You can think of this as if we
      // were scanning the sequences from the shortest one, and every time we realize there's
      // more elements below in our column, we lower the counter (prev_l), and append the new
      // block to the output.
      i64 prev_l = 0;
      for (i64 i = 0; i < batch_size; ++i) {
        i64 l = lengths[batch_size - 1 - i];
        if (l > prev_l) {
          auto current_batch_size = batch_size - i;
          steps.push_back(input.slice(0, prev_l, l).slice(1, 0, current_batch_size).contiguous().view(step_shape));
          for (i64 j = 0; j < (l - prev_l); ++j) {
            (*batch_sizes++) = current_batch_size;
          }
          prev_l = l;
        }
        TORCH_CHECK(l >= prev_l);
      }

      return make_tuple(cat(steps), batch_sizes_t);
        */
}

/**
  | `grad` could be on arbitrary device and of
  | arbitrary dtype, but `_batch_sizes` is
  | guaranteed to be a CPU int64 tensor.
  |
  | See NOTE [ device and dtype of a PackedSequence ]
  */
pub fn pack_padded_sequence_backward(
        grad:        &Tensor,
        input_size:  &[i32],
        batch_sizes: &Tensor,
        batch_first: bool) -> Tensor {
    
    todo!();
        /*
            vector<i64> input_size_after_t = input_size.vec();
      if (batch_first) {
        TORCH_CHECK(input_size.size() >= 2);
        swap(input_size_after_t[0], input_size_after_t[1]);
      }
      auto grad_input = zeros(input_size_after_t, grad.options());
      auto batch_sizes_t = _batch_sizes.contiguous();
      checkLongTensor(batch_sizes_t);

      i64 offset = 0;
      i64 max_seq_len = batch_sizes_t.size(0);
      i64 * batch_sizes = batch_sizes_t.data_ptr<i64>();
      for (i64 i = 0; i < max_seq_len; ++i) {
        grad_input[i].slice(0, 0, batch_sizes[i]).copy_(grad.slice(0, offset, offset + batch_sizes[i]));
        offset += batch_sizes[i];
      }

      if (batch_first) {
        grad_input = grad_input.transpose(0, 1);
      }

      return grad_input;
        */
}

pub fn pad_packed_sequence(
        data:          &Tensor,
        batch_sizes:   &Tensor,
        batch_first:   bool,
        padding_value: &Scalar,
        total_length:  i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
            auto batch_sizes_t = _batch_sizes.contiguous();
      checkLongTensor(batch_sizes_t);

      i64 * batch_sizes = batch_sizes_t.data_ptr<i64>();
      i64 max_batch_size = batch_sizes[0];
      i64 max_real_seq_length = batch_sizes_t.size(0);
      i64 max_seq_length = max_real_seq_length;
      if (total_length > 0) {
        TORCH_CHECK(total_length >= max_seq_length,
                 "Expected total_length to be at least the length of the longest "
                 "sequence in input, but got total_length=", total_length, " and "
                 "max sequence length being ", max_seq_length);
        max_seq_length = total_length;
      }

      vector<i64> output_size; // == [max_seq_length, max_batch_size, *var_data.size()[1:]]
      {
        output_size.reserve(data.dim() + 1);
        output_size.push_back(max_seq_length);
        output_size.push_back(max_batch_size);
        auto s_data_size = data.sizes().slice(1);
        output_size.insert(output_size.end(), s_data_size.begin(), s_data_size.end());
      }
      auto output = full(output_size, padding_value, data.options());

      // This will be modified at every iteration, but we reserve memory for it now.
      vector<i64> tmp_view_size = move(output_size); // == [-1, -1, *var_data.size()[1:]]

      Tensor lengths_t = empty(max_batch_size, batch_sizes_t.options());
      i64 * lengths = lengths_t.data_ptr<i64>() + max_batch_size - 1;
      i64 data_offset = 0;
      i64 prev_batch_size = max_batch_size;
      i64 prev_i = 0;
      for (i64 i = 0; i <= max_real_seq_length; ++i) {
        i64 batch_size = i != max_real_seq_length ? batch_sizes[i] : 0;
        if (batch_size != prev_batch_size) {
          i64 l = prev_batch_size * (i - prev_i);
          // The lines below are equivalent to this:
          // output[prev_i:i, :prev_batch_size] = tmp.view(i - prev_i, prev_batch_size, *input.shape[2:])
          auto tmp = data.slice(0, data_offset, data_offset + l);
          tmp_view_size[0] = i - prev_i;
          tmp_view_size[1] = prev_batch_size;
          output.slice(0, prev_i, i).slice(1, 0, prev_batch_size).copy_(tmp.view(tmp_view_size));
          data_offset += l;
          prev_i = i;
        }
        i64 dec = prev_batch_size - batch_size;
        if (dec > 0) {
          for (i64 j = 0; j < dec; ++j) {
            (*lengths--) = i;
          }
        }
        prev_batch_size = batch_size;
      }

      if (batch_first) {
        output = output.transpose(0, 1);
      }

      return make_tuple(output, lengths_t);
        */
}

pub fn pad_sequence(
        sequences:     &[Tensor],
        batch_first:   bool,
        padding_value: f64) -> Tensor {
    
    todo!();
        /*
            const i64 sequences_size = sequences.size();
      TORCH_CHECK(sequences_size > 0, "received an empty list of sequences");
      IntArrayRef max_size = sequences[0].sizes();
      IntArrayRef trailing_dims = max_size.slice(1);
      i64 max_len = max_element(
        sequences.begin(),
        sequences.end(),
        [](const Tensor &a, const Tensor &b) {
          return a.size(0) < b.size(0);
        }
      )->size(0);

      DimVector out_dims;
      if (batch_first) {
        out_dims = {sequences_size, max_len};
      } else {
        out_dims = {max_len, sequences_size};
      }
      out_dims.insert(out_dims.end(), trailing_dims.begin(), trailing_dims.end());

      Tensor out = full(out_dims, padding_value, sequences[0].options());
      for (i64 i = 0; i < sequences_size; i++) {
        const Tensor currseq = sequences[i];
        const i64 length_i = currseq.size(0);
        // use index notation to prevent duplicate references to the tensor
        if (batch_first) {
          out.select(0, i).narrow(0, 0, length_i).copy_(currseq);
        } else {
          out.narrow(0, 0, length_i).select(1, i).copy_(currseq);
        }
      }
      return out;
        */
}
