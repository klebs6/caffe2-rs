crate::ix!();

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
#[inline] pub fn compute_input_size_(inputs: &Vec<IValue>) -> i32 {
    
    todo!();
    /*
        if (inputs.empty()) {
        return 0;
      }
      if (inputs[0].isTensorList()) {
        // if the first input is a tensor list, we get input tensors by indexing
        // into that list. currently, this means that only tensors from that list
        // are accessible as inputs. any hypothetical input tensors that come after
        // the list are not accessible.
        return inputs[0].toTensorVector().size();
      }
      // it's not a tensor list. Count the number of tensor inputs and return them.
      size_t num_tensor_inputs = 0;
      bool found_nontensor = false;
      for (const auto& input : inputs) {
        if (input.isTensor()) {
          AT_ASSERTM(
              !found_nontensor,
              "All tensor arguments must come before non-tensor arguments");
          ++num_tensor_inputs;
        } else {
          found_nontensor = true;
        }
      }
      return num_tensor_inputs;
    */
}
