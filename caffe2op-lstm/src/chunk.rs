crate::ix!();

#[inline] pub fn chunk(
    input:   &Tensor,
    chunks:  i32,
    axis:    i32,
    context: *mut CPUContext) -> Vec<Tensor> 
{
    todo!();
    /*
        int canonical_axis = input.canonical_axis_index(axis);
      CAFFE_ENFORCE_LT(
          canonical_axis, input.dim(), "Axis not in input ndim range.");
      const int input_channels = input.dim32(canonical_axis);
      CAFFE_ENFORCE_EQ(
          input_channels % chunks,
          0,
          "input channels should be divisible by the number of chunks.");
      auto split_size = input_channels / chunks;
      vector<int64_t> output_dims(input.sizes().vec());
      int before = 1, after = 1;
      for (int i = 0; i < canonical_axis; ++i) {
        before *= input.dim32(i);
      }
      for (int i = canonical_axis + 1; i < input.dim(); ++i) {
        after *= input.dim32(i);
      }
      size_t input_offset = 0;
      std::vector<Tensor> outputs;
      for (int i = 0; i < chunks; ++i) {
        auto axis_dim = split_size;
        output_dims[canonical_axis] = split_size;
        Tensor output(output_dims, CPU);
        math::CopyMatrix<CPUContext>(
            input.itemsize(),
            before,
            axis_dim * after,
            static_cast<const char*>(input.raw_data()) + input_offset,
            input.dim32(canonical_axis) * after,
            output.raw_mutable_data(input.dtype()),
            axis_dim * after,
            context,
            input.dtype().copy());
        input_offset += axis_dim * after * input.itemsize();
        outputs.push_back(std::move(output));
      }
      return outputs;
    */
}
