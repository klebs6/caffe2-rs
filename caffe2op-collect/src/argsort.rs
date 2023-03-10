crate::ix!();

/**
  | Updates arr to be indices that would
  | sort the array. Implementation of https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
  |
  */
#[inline] pub fn arg_sort(arr: &mut EArrXi)  {
    
    todo!();
    /*
        // Create index array with 0, 1, ... N and sort based on array values
      std::vector<int> idxs(arr.size());
      std::iota(std::begin(idxs), std::end(idxs), 0);
      std::sort(idxs.begin(), idxs.end(), [&arr](int lhs, int rhs) {
        return arr(lhs) < arr(rhs);
      });
      // Update array to match new order
      for (int i = 0; i < arr.size(); i++) {
        arr(i) = idxs[i];
      }
    */
}
