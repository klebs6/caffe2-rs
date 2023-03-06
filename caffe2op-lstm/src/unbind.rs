crate::ix!();

#[inline] pub fn unbind(
    input:   &Tensor,
    axis:    i32,
    context: *mut CPUContext) -> Vec<Tensor> 
{
    
    todo!();
    /*
        // 1 - Chunk the input tensor along the given axis into N chunks where
      // N is the dim(axis)
      auto chunks = chunk(input, input.sizes()[axis], axis, context);
      // 2 - Compute new dimensions
      std::vector<int64_t> newDims = input.sizes().vec();
      newDims.erase(newDims.begin() + axis);

      // 3 - Reshape chunks to drop the extra dimension
      for (int i = 0; i < chunks.size(); i++) {
        CAFFE_ENFORCE_EQ(
            chunks[i].sizes()[axis], 1, "Got an unexpected chunk size");
        chunks[i].Reshape(newDims);
      }
      return chunks;
    */
}
