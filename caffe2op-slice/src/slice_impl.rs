crate::ix!();

#[inline] pub fn slice_impl<SIndex, Context>(
    output:  *mut Tensor,
    data:    &Tensor,
    starts:  &Tensor,
    ends:    &Tensor,
    context: *mut Context,
    gdata:   Option<&mut Tensor>,
    go:      Option<&Tensor>) -> bool {

    todo!();
    /*
        bool backward = output == nullptr;

      auto* starts_data = starts.template data<SIndex>();
      auto* ends_data = ends.template data<SIndex>();

      CAFFE_ENFORCE_EQ(starts.dim(), 1);
      CAFFE_ENFORCE_EQ(ends.dim(), 1);
      CAFFE_ENFORCE_GE(data.dim(), starts.numel());
      CAFFE_ENFORCE_EQ(starts.numel(), ends.numel());

      std::vector<SIndex> starts_idx(data.dim());
      std::vector<SIndex> ends_idx(data.dim());
      std::vector<SIndex> dst_sizes(data.dim());

      for (int i = 0; i < data.dim(); ++i) {
        if (i >= starts.numel()) {
          starts_idx[i] = 0;
          ends_idx[i] = data.size(i);
          dst_sizes[i] = data.size(i);
          continue;
        }
        if (data.size(i) > 0) {
          auto start = starts_data[i];
          auto end = ends_data[i];
          if (start < 0) {
            start = data.size(i) + 1 + start;
          }
          if (end < 0) {
            end = data.size(i) + 1 + end;
          }
          if (start > data.size(i)) {
            start = data.size(i);
          }
          if (end > data.size(i)) {
            end = data.size(i);
          }
          CAFFE_ENFORCE_GE(start, 0);
          CAFFE_ENFORCE_GE(end, 0);
          CAFFE_ENFORCE_GE(end, start);
          starts_idx[i] = start;
          ends_idx[i] = end;
          dst_sizes[i] = end - start;
        } else {
          starts_idx[i] = 0;
          ends_idx[i] = 0;
          dst_sizes[i] = 0;
        }
      }

      if (data.numel() <= 0) {
        // When the input is empty, we do not need to do copy.
        if (!backward) {
          output->Resize(dst_sizes);
          output->raw_mutable_data(data.dtype());
        } else {
          gdata->ResizeLike(data);
          gdata->raw_mutable_data(go->dtype());
        }
        return true;
      }
      // for now only supports slicing in 1 dimension
      int dim = -1;
      for (int i = 0; i < data.dim(); ++i) {
        if (starts_idx[i] > 0 || ends_idx[i] < data.size(i)) {
          CAFFE_ENFORCE_EQ(
              dim, -1, "Currently only possible to slice in 1 dimension.");
          dim = i;
        }
      }
      if (dim == -1) {
        if (!backward) {
          output->CopyFrom(data, true /*async*/);
        } else {
          gdata->CopyFrom(*go, true /*async*/);
        }
        return true;
      }
      size_t unit = std::accumulate(
          data.sizes().begin() + dim + 1,
          data.sizes().end(),
          1,
          std::multiplies<SIndex>());
      size_t num_blocks = std::accumulate(
          data.sizes().begin(),
          data.sizes().begin() + dim,
          1,
          std::multiplies<SIndex>());
      if (!backward) {
        output->Resize(dst_sizes);
      } else {
        gdata->ResizeLike(data);
      }

      size_t itemsize = data.dtype().itemsize();

      if (!backward) {
        char* src_bytes = (char*)data.raw_data();
        char* dst_bytes = (char*)output->raw_mutable_data(data.dtype());

        size_t src_nbytes = data.nbytes();
        size_t dst_nbytes = output->nbytes();

        size_t src_block_size = unit * data.size(dim);
        size_t dst_block_size = unit * (ends_idx[dim] - starts_idx[dim]);
        size_t src_offset = unit * starts_idx[dim];

        if (num_blocks == 0 || dst_block_size == 0) {
          return true;
        }

        size_t src_block_size_bytes = itemsize * src_block_size;
        size_t dst_block_size_bytes = itemsize * dst_block_size;

        char* src_offset_bytes = src_bytes + itemsize * src_offset;
        char* dst_offset_bytes = dst_bytes;
        for (size_t i = 0; i < num_blocks; ++i) {
          char* local_src_offset_bytes =
              src_offset_bytes + i * src_block_size_bytes;
          char* local_dst_offset_bytes =
              dst_offset_bytes + i * dst_block_size_bytes;
          DCHECK_LE(
              static_cast<void*>(local_src_offset_bytes + dst_block_size_bytes),
              static_cast<void*>(src_bytes + src_nbytes));
          DCHECK_LE(
              static_cast<void*>(local_dst_offset_bytes + dst_block_size_bytes),
              static_cast<void*>(dst_bytes + dst_nbytes));
          context->CopyItemsSameDevice(
              data.dtype(),
              dst_block_size,
              (void*)local_src_offset_bytes,
              (void*)local_dst_offset_bytes);
        }
      } else {
        char* src_bytes = (char*)go->raw_data();
        char* dst_bytes = (char*)gdata->raw_mutable_data(go->dtype());

        size_t src_nbytes = go->nbytes();
        size_t dst_nbytes = gdata->nbytes();

        size_t src_block_size = unit * (ends_idx[dim] - starts_idx[dim]);
        size_t dst_block_size = unit * data.size(dim);
        size_t dst_offset = unit * starts_idx[dim];

        if (num_blocks == 0 || dst_block_size == 0) {
          return true;
        }

        size_t src_block_size_bytes = itemsize * src_block_size;
        size_t dst_block_size_bytes = itemsize * dst_block_size;

        char* src_offset_bytes = src_bytes;
        char* dst_offset_bytes = dst_bytes + itemsize * dst_offset;
        // Zero out gradient blob before copy since we copy in fewer items than
        // there is space for
        math::Set<char, Context>(dst_nbytes, 0, dst_bytes, context);

        // If output tensor is empty, just return zeroed gradient tensor
        if (!src_bytes) {
          return true;
        }

        for (size_t i = 0; i < num_blocks; ++i) {
          char* local_src_offset_bytes =
              src_offset_bytes + i * src_block_size_bytes;
          char* local_dst_offset_bytes =
              dst_offset_bytes + i * dst_block_size_bytes;
          DCHECK_LE(
              local_src_offset_bytes + src_block_size_bytes,
              src_bytes + src_nbytes);
          DCHECK_LE(
              local_dst_offset_bytes + src_block_size_bytes,
              dst_bytes + dst_nbytes);
          context->CopyItemsSameDevice(
              go->dtype(),
              src_block_size,
              (void*)local_src_offset_bytes,
              (void*)local_dst_offset_bytes);
        }
      }
      return true;
    */
}
