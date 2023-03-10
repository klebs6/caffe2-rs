crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/AutogradComposite.cpp]

/**
  | This function can be used to create a dual
  | Tensor that holds a tangent to compute forward
  | mode gradients.
  |
  | Note that the dual Tensor's primal is a view
  | of the given primal and the given tangent is
  | used as-is.
  |
  | This function is backward differentiable.
  */
pub fn make_dual(
        primal:  &Tensor,
        tangent: &Tensor,
        level:   i64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!primal._fw_grad(level).defined(), "Making a dual Tensor based on a Tensor that "
                  "already has a forward gradient at the same level ", level, " is not supported.");

      auto dual_tensor = primal.view(primal.sizes());
      dual_tensor._set_fw_grad(tangent, level, /* is_inplace_op */ false);
      return dual_tensor;
        */
}

/**
  | This function can be used to unpack a given
  | dual Tensor to get its primal and tangent. The
  | returned primal is a view of the dual and the
  | tangent is returned as is.
  |
  | This function is backward differentiable.
  */
pub fn unpack_dual(
        tensor: &Tensor,
        level:  i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return tuple<Tensor, Tensor>(tensor._fw_primal(level), tensor._fw_grad(level));
        */
}
