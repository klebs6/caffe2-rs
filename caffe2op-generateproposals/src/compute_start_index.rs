crate::ix!();

/**
 | Compute the 1-d index of a n-dimensional
 | contiguous row-major tensor for a given
 | n-dimensional index 'index'
 |
 */
#[inline] pub fn compute_start_index(
    tensor: &TensorCPU,
    index:  &Vec<i32>) -> usize 
{
    todo!();
    /*
        DCHECK_EQ(index.size(), tensor.dim());

      size_t ret = 0;
      for (int i = 0; i < index.size(); i++) {
        ret += index[i] * tensor.size_from_dim(i + 1);
      }

      return ret;
    */
}
