crate::ix!();

/**
  | Actual gather implementation - resizes
  | output and copies indexed data.
  |
  */
#[inline] pub fn gather_impl<Index, Context>(
    op:            *mut dyn Operator,
    data_idx:      i32,
    indices_idx:   i32,
    output_idx:    i32,
    axis:          i32,
    wrap_indices:  bool,
    match_outer:   bool) -> bool 
{
    todo!();
    /*
        // If we endup using it on GPU doing O(N) memcpy is probably not best :)
      // TODO: implement prefetching if it starts mattering (TF does it)

      const Tensor& data = op->Input(dataIdx);
      const Tensor& indices = op->Input(indicesIdx);
      const TypeMeta dataType = data.dtype();
      size_t item_bytesize = dataType.itemsize();

      // ONNX allows negative axis to index from the back, valid range: [-r, r].
      if (axis < 0) {
        axis = data.dim() + axis;
      }
      CAFFE_ENFORCE_GE(data.dim(), axis + 1, "DATA should be at least [axis+1]-D");
      CAFFE_ENFORCE_GE(axis, 0, "Axis should be non-negative");
      CAFFE_ENFORCE_LT(axis, data.dim(), "Axis out of range");

      // New shape:
      //  [data dims before axis] + [indices dims] + [data dims after axis]
      vector<int64_t> shape = calc_output_shape_vector<int64_t>(
          data.sizes(), indices.sizes(), axis, match_outer);
      Tensor* output = op->Output(outputIdx, shape, at::dtype(dataType));
      auto out = static_cast<char*>(output->raw_mutable_data(dataType));

      // Succeed if size of output is zero, which can happen for empty batch which
      // would have data dimension size of 0.
      // This *must* be done AFTER output->raw_mutable_data() above as that has
      // important allocation side effect that we must see.
      if (output->numel() == 0) {
        return true;
      }

      const Index* idxs = indices.template data<Index>();
      auto src_base = static_cast<const char*>(data.raw_data());

      auto outer_dims_product = data.size_to_dim(axis);
      auto block_size = data.size_from_dim(axis + 1);
      auto block_bytesize = block_size * item_bytesize;

      auto src_indexing_axis_dim = data.size(axis);
      auto src_batch_bytesize = data.size_from_dim(axis) * item_bytesize;
      // Treat indices as a single block even if they have multiple dimensions.
      // The "gathered batch" is a cumulative result combining indexed blocks.
      auto idx_inner_dims_product = indices.size_from_dim(axis);
      auto N = indices.numel();
      if (match_outer) {
        CAFFE_ENFORCE_GE(axis, 1, "Axis should be at least 1");
        for (auto i = 0; i < axis; i++) {
          CAFFE_ENFORCE_EQ(
              data.size(i),
              indices.size(i),
              "INDICES must have the same outer dims as DATA (before dim AXIS)");
        }
        N = idx_inner_dims_product;
      }

      auto gathered_batch_bytesize = N * block_size * item_bytesize;

      check_indexarray_range<Index>(idxs, N, src_indexing_axis_dim, wrap_indices);

      // Special-case single-float copy for efficiency
      if (data.template IsType<float>() && block_size == 1) {
        for (auto batch = 0; batch < outer_dims_product; ++batch) {
          const float* src_floats =
              (const float*)(src_base + batch * src_batch_bytesize);
          float* dst_floats = (float*)(out + batch * gathered_batch_bytesize);

          for (auto i = 0; i < N; ++i) {
            auto idx = idxs[i];
            if (match_outer) {
              idx = idxs[batch * idx_inner_dims_product + i];
            }
            if (wrap_indices && idx < 0) {
              idx = idx + src_indexing_axis_dim;
            }
            dst_floats[i] = src_floats[idx];
          }
        }
      } else {
        // outer_dims_product specifies how many times we repeat inner dimensions,
        // so we just iterate over it to cover all outer dimensions.
        for (auto batch = 0; batch < outer_dims_product; ++batch) {
          for (auto i = 0; i < N; ++i) {
            auto idx = idxs[i];
            if (match_outer) {
              idx = idxs[batch * idx_inner_dims_product + i];
            }
            if (wrap_indices && idx < 0) {
              idx = idx + src_indexing_axis_dim;
            }

            auto src = src_base + batch * src_batch_bytesize + idx * block_bytesize;
            auto dst = out + batch * gathered_batch_bytesize + i * block_bytesize;
            op->getContext()->CopyItemsSameDevice(dataType, block_size, src, dst);
          }
        }
      }
      return true;
    */
}
