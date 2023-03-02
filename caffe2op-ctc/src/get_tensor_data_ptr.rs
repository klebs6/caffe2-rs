crate::ix!();

#[inline] pub fn get_tensor_data_ptr(
    tensor: &Tensor, 
    t: i32, 
    n: i32) -> *const f32 
{
    todo!();
    /*
        const auto dims = tensor.sizes();
      CAFFE_ENFORCE_EQ(dims.size(), 3);
      int offset = (t * dims[1] + n) * dims[2];
      CAFFE_ENFORCE_LT(offset, tensor.numel());
      return tensor.template data<float>() + offset;
    */
}
