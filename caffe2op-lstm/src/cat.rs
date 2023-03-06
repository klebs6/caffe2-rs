crate::ix!();

#[inline] pub fn cat(
    tensor_list: &Vec<Tensor>,
    axis:        i32,
    context:     *mut CPUContext) -> Tensor 
{
    todo!();
    /*
        // Adopted from C2's concat operator
      auto input_zero = copy_ctor(tensorList.at(0));
      vector<int64_t> outputDims(input_zero.sizes().vec());
      CAFFE_ENFORCE(outputDims.size() > 0);
      for (int i = 1; i < tensorList.size(); i++) {
        CAFFE_ENFORCE(input_zero.dtype() == tensorList.at(i).dtype());
        outputDims[axis] += tensorList.at(i).sizes()[axis];
      }
      auto output_channels = outputDims[axis];
      Tensor output(outputDims, CPU);
      int before = 1, after = 1;
      for (int i = 0; i < tensorList.at(0).dim(); ++i) {
        if (i == axis) {
          continue;
        }
        int dim = input_zero.dim32(i);
        if (i < axis) {
          before *= dim;
        } else {
          after *= dim;
        }
      }
      size_t output_offset = 0;
      for (const auto& input : tensorList) {
        auto axis_dim = input.dim32(axis);
        math::CopyMatrix<CPUContext>(
            input.itemsize(),
            before,
            axis_dim * after,
            input.raw_data(),
            axis_dim * after,
            static_cast<char*>(output.raw_mutable_data(input_zero.dtype())) +
                output_offset,
            output_channels * after,
            context,
            input_zero.dtype().copy());
        output_offset += axis_dim * after * input.itemsize();
      }

      return output;
    */
}
