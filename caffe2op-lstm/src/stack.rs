crate::ix!();

#[inline] pub fn stack(
    tensor_list: &Vec<Tensor>,
    axis:        i32,
    context:     *mut CPUContext) -> Tensor 
{
    
    todo!();
    /*
        // 1 - Compute new dimensions
      std::vector<int64_t> newDims(tensorList[0].sizes().vec());
      std::vector<Tensor> expandedTensorList;
      newDims.insert(newDims.begin() + axis, 1);
      for (int i = 0; i < tensorList.size(); i++) {
        expandedTensorList.emplace_back(tensorList[i].Clone());
        expandedTensorList.at(i).Reshape(newDims);
      }
      return cat(expandedTensorList, axis, context);
    */
}
