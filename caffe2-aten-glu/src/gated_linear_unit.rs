crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/GatedLinearUnit.cpp]

define_dispatch!{glu_stub}
define_dispatch!{glu_backward_stub}

pub fn glu_out<'a>(
        self_:  &Tensor,
        dim:    i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
      // can't be evenly halved, but give a nicer error message here.
      TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
      auto wrap_dim = maybe_wrap_dim(dim, self.dim());
      const i64 nIn = self.size(wrap_dim);
      TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
                  wrap_dim, " is size ", nIn);
      // size output to half of input
      const i64 selfSize = nIn / 2;
      auto newSizes = self.sizes().vec();
      newSizes[wrap_dim] = selfSize;
      result.resize_(newSizes);
      // half tensor
      Tensor firstHalf = self.narrow(wrap_dim, 0, selfSize);
      Tensor secondHalf = self.narrow(wrap_dim, selfSize, selfSize);

      auto iter = TensorIterator::borrowing_binary_op(result, firstHalf, secondHalf);
      glu_stub(iter.device_type(), iter);
      return result;
        */
}

pub fn glu(
        self_: &Tensor,
        dim:   i64) -> Tensor {
    
    todo!();
        /*
            auto result = empty({0}, self.options());
      return glu_out(result, self, dim);
        */
}

pub fn glu_backward_out<'a>(
        grad_output: &Tensor,
        input:       &Tensor,
        dim:         i64,
        grad_input:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(input.dim() > 0, "glu does not support 0-dimensional tensors");
      auto wrap_dim = maybe_wrap_dim(dim, input.dim());
      const i64 nIn = input.size(wrap_dim);
      TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
                  wrap_dim, " is size ", nIn);

      grad_input.resize_as_(input);
      const i64 inputSize = nIn / 2;
      // half tensor
      Tensor firstHalf = input.narrow(wrap_dim, 0, inputSize);
      Tensor secondHalf = input.narrow(wrap_dim, inputSize, inputSize);
      Tensor gradInputfirstHalf = grad_input.narrow(wrap_dim, 0, inputSize);
      Tensor gradInputsecondHalf = grad_input.narrow(wrap_dim, inputSize, inputSize);

      sigmoid_out(gradInputfirstHalf, secondHalf);
      // for second gradinput half, can get a better performance by fusion
      auto iter = TensorIteratorConfig()
        .add_output(gradInputsecondHalf)
        .add_input(gradInputfirstHalf)
        .add_input(firstHalf)
        .add_input(grad_output)
        .build();
      glu_backward_stub(iter.device_type(), iter);
      gradInputfirstHalf.mul_(grad_output);
      return grad_input;
        */
}

pub fn glu_backward(
        grad_output: &Tensor,
        input:       &Tensor,
        dim:         i64) -> Tensor {
    
    todo!();
        /*
            auto grad_input = empty({0}, input.options());
      return glu_backward_out(grad_input, grad_output, input, dim);
        */
}
