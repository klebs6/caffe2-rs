crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/TensorCompare.cpp]

pub fn max_quantized_cpu(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return get<0>(self.reshape({-1}).max(/*dim=*/0));
        */
}


pub fn min_quantized_cpu(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return get<0>(self.reshape({-1}).min(/*dim=*/0));
        */
}

/**
  | TODO: move to TensorMath.cpp
  |
  */
pub fn sort_quantized_cpu_stable(
        self_:      &Tensor,
        stable:     Option<bool>,
        dim:        i64,
        descending: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            Tensor sort_int;
      Tensor sort_indicies;
      tie(sort_int, sort_indicies) =
          sort(self.int_repr(), stable, dim, descending);
      return forward_as_tuple(
          _make_per_tensor_quantized_tensor(
              sort_int, self.q_scale(), self.q_zero_point()),
          sort_indicies);
        */
}

pub fn sort_quantized_cpu(
        self_:      &Tensor,
        dim:        i64,
        descending: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return sort_quantized_cpu_stable(self, /*stable=*/false, dim, descending);
        */
}
