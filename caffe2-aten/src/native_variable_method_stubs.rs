/*!
  | The stubs in here are used by dynamic
  | dispatch. It just redirects everything to the
  | Tensor method we manually bind in TensorBody.h.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/VariableMethodStubs.cpp]

pub fn backward(
    self_:        &Tensor,
    inputs:       TensorList,
    gradient_opt: &Option<Tensor>,
    keep_graph:   Option<bool>,
    create_graph: bool)  {

    todo!();
        /*
            return self._backward(inputs, gradient_opt, keep_graph, create_graph);
        */
}

pub fn set_data(
        self_:    &mut Tensor,
        new_data: &Tensor)  {
    
    todo!();
        /*
            return self.set_data(new_data);
        */
}

pub fn data(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.data();
        */
}

pub fn is_leaf(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return self.is_leaf();
        */
}

pub fn output_nr(self_: &Tensor) -> i64 {
    
    todo!();
        /*
            return self.output_nr();
        */
}

pub fn version(self_: &Tensor) -> i64 {
    
    todo!();
        /*
            return self._version();
        */
}

pub fn requires_grad(
        self_:         &mut Tensor,
        requires_grad: bool) -> &mut Tensor {
    
    todo!();
        /*
            self.requires_grad_(_requires_grad);
      return self;
        */
}

pub fn retain_grad(self_: &mut Tensor)  {
    
    todo!();
        /*
            return self.retain_grad();
        */
}

pub fn retains_grad(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return self.retains_grad();
        */
}

pub fn fw_primal(
        self_: &Tensor,
        level: i64) -> Tensor {
    
    todo!();
        /*
            AT_ERROR("_fw_primal is not implemented for Tensor");
        */
}
