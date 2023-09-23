crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/sparse/ParamUtils.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/sparse/ParamUtils.cpp]

pub fn softmax_sparse_input_preprocessing(
        input:         &Tensor,
        dim:           i64,
        half_to_float: bool,
        function_name: CheckedFrom) -> (Tensor,Tensor) {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(input_.is_sparse());
      TORCH_CHECK(
          !half_to_float,
          string(function_name) +
              ": with half to float conversion is not supported on " +
              input_.device().str());
      auto input = input_.coalesce();
      Tensor output = native::empty_like(input);
      TORCH_CHECK(
          dim_ >= 0 && dim_ < input.dim(),
          ": dim must be non-negative and less than input dimensions");
      return make_pair(input, output);
        */
}

pub fn softmax_backward_sparse_input_preprocessing(
        grad:          &Tensor,
        output:        &Tensor,
        dim:           i64,
        input:         &Tensor,
        function_name: CheckedFrom) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            TensorArg grad_arg{grad_, "grad", 1}, output_arg{output_, "output", 2};
      checkSameSize(function_name, grad_arg, output_arg);

      i64 dim = maybe_wrap_dim(dim_, grad_.dim());

      auto grad = grad_.coalesce();
      auto output = output_.coalesce();

      Tensor grad_input = native::empty_like(output);
      TORCH_CHECK(
          dim >= 0 && dim < grad.dim(),
          ": dim must be non-negative and less than input dimensions");
      TORCH_CHECK(
          grad.sparse_dim() == output.sparse_dim(),
          ": grad and output sparse dimensions must be equal");
      return make_tuple(grad_input, grad, output);
        */
}
