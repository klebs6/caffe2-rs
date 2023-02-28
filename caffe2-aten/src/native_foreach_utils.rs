/*!
  | To go via 'fast' path, several conditions must
  | be satisfied
  |
  | - All tensors in all lists must have the same
  | dtype.
  |
  | - All tensors must be on the same device
  |
  | - All tensors must have strided layout
  |
  | - All tensors must be non-overlapping and dense
  |
  | - Resulting tensor must have the same dtype as
  | the input one
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ForeachUtils.h]

/**
  | Check if tensor list has either a boolean
  | tensor or a integer tensor
  |
  */
pub fn has_integral_tensor(
        tensors:      TensorList,
        include_bool: bool) -> bool {
    
    todo!();
        /*
            return any_of(tensors.begin(), tensors.end(),
        [&includeBool](const auto & t) { return isIntegralType(t.scalar_type(), includeBool); });
        */
}

/// check if tensor list has bool tensors
///
pub fn has_bool_tensor(tensors: TensorList) -> bool {
    
    todo!();
        /*
            return any_of(tensors.begin(), tensors.end(),
        [](const auto & t) -> bool { return t.scalar_type() == ScalarType::Bool; });
        */
}

/**
  | Check foreach API restrictions
  |
  | - Tensor lists must be non-empty.
  |
  | - All TensorLists and ScalarLists must have the
  | same number of elements.
  |
  | - Corresponding tensors must have the same
  | size.
  |
  */
pub fn check_foreach_api_restrictions_a(tensors: TensorList)  {
    
    todo!();
        /*
            TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
        */
}

pub fn check_foreach_api_restrictions_b(
        tensors: TensorList,
        scalars: &[Scalar])  {
    
    todo!();
        /*
            check_foreach_api_restrictions(tensors);
      TORCH_CHECK(tensors.size() == scalars.size(), "Tensor list must have same number of elements as scalar list.");
        */
}

pub fn check_foreach_api_restrictions_c(
        tensors1: TensorList,
        tensors2: TensorList)  {
    
    todo!();
        /*
            TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
      TORCH_CHECK(tensors2.size() > 0, "Tensor list must have at least one tensor.");
      TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors2.size());

      for (const auto i : irange(tensors1.size())) {
        TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors2[i].sizes());
      }
        */
}

pub fn check_foreach_api_restrictions_d(
        tensors1: TensorList,
        tensors2: TensorList,
        tensors3: TensorList)  {
    
    todo!();
        /*
            TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
      TORCH_CHECK(tensors2.size() > 0, "Tensor list must have at least one tensor.");
      TORCH_CHECK(tensors3.size() > 0, "Tensor list must have at least one tensor.");
      TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors2.size());
      TORCH_CHECK(tensors1.size() == tensors3.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors3.size());

      for (const auto i : irange(tensors1.size())) {
        TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors2[i].sizes());
        TORCH_CHECK(tensors1[i].sizes() == tensors3[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors3[i].sizes());
      }
        */
}

pub fn check_foreach_api_restrictions_e(
        tensors1: TensorList,
        tensors2: TensorList,
        tensors3: TensorList,
        scalars:  &[Scalar])  {
    
    todo!();
        /*
            check_foreach_api_restrictions(tensors1, tensors2, tensors3);
      TORCH_CHECK(tensors1.size() == scalars.size(), "Tensor list must have same number of elements as scalar list, got ", tensors1.size(), " and ", scalars.size());
        */
}


/**
  | TODO(mkozuki): Consider whether we really need
  | this function or not.
  |
  | Note that, there is a possibility that foreach
  | fastpath supports type promotion in the future,
  | which might complicate the functionality this
  | function should provides.
  |
  | However, as of now, the check of division op
  | with integer inputs is
  | duplicated. `check_fast_path_restrictions` does
  | the same thing in it before calling this
  | function.
  |
  */
pub fn will_promote_tensor(
    tensor:                                  &Tensor,
    scalar:                                  &Scalar,
    does_op_promote_integer_inputs_to_float: bool) -> bool {

    let does_op_promote_integer_inputs_to_float: bool = does_op_promote_integer_inputs_to_float.unwrap_or(false);

    todo!();
        /*
            // In case of division, integer inputs will result in float
      if (does_op_promote_integer_inputs_to_float &&
          isIntegralType(tensor.scalar_type(), /* includeBool */ true)) {
        return true;
      }
      return tensor.scalar_type() != native::result_type(scalar, tensor);
        */
}

/**
  | Please, make sure to call
  | check_foreach_api_restrictions before calling
  | this method.
  |
  | There is a set of preconditions that have to be
  | satisfied.
  |
  */
pub fn check_fast_path_restrictions(
    tensor_lists:                            &[TensorList],
    scalar_list:                             &[Scalar],
    does_op_promote_integer_inputs_to_float: bool) -> bool {

    let does_op_promote_integer_inputs_to_float: bool = does_op_promote_integer_inputs_to_float.unwrap_or(false);

    todo!();
        /*
            const auto expected_dtype = tensorLists[0][0].dtype();
        const auto expected_device = tensorLists[0][0].device();

        auto is_tensor_okay = [&](const Tensor& tensor) {
          return tensor.dtype() == expected_dtype &&
                 tensor.device() == expected_device &&
                 tensor.layout() == kStrided &&
                 tensor.is_non_overlapping_and_dense();
        };

        for (const auto& tensorList : tensorLists) {
          for (const auto& tensor : tensorList) {
            if (!is_tensor_okay(tensor)) {
              return false;
            }
          }
        }

        // Check if corresponding tensors in tensor lists have the same strides.
        for (int i=0; i < tensorLists.size(); i++) {
          for (int j=0; j < tensorLists[0].size(); j++) {
            if (tensorLists[0][j].strides() != tensorLists[i][j].strides()) {
              return false;
            }
          }
        }

        // This function has already checked that `tensorList[j][i]` for all j, i has the same dtype
        // using `is_tensor_okay` function above.
        // checked by `check_foreach_api_restrictions`).
        // This means we only need to check if {tensorList[0][0], tensorList[0][1], tensorList[0][2], ...}
        // do type promotion with scalarLIst.
        for (int i=0; i < tensorLists[0].size(); i++) {
          if (does_op_promote_integer_inputs_to_float) {
            if (isIntegralType(tensorLists[0][i].scalar_type(), /*includeBool*/ true)) {
              return false;
            }
          }

          if (scalarList.size() == 1) {
            if (will_promote_tensor(tensorLists[0][i], scalarList[0])) {
              return false;
            }
          } else if (scalarList.size() > 1) {
            // FIXME(mkozuki): Consider specializing `TensorListScalarListMetadata` for complex dtypes
            // to access the following comment.
            // Complex scalar list is not supported due to the limit for kernel launch argument (4KB)
            if (scalarList[i].isComplex()) {
              return false;
            }

            if (will_promote_tensor(tensorLists[0][i], scalarList[i])) {
              return false;
            }
          }
        }

        return true;
        */
}

pub fn can_use_fast_route_a(
    tensor_lists:                            &[TensorList],
    scalar_list:                             &[Scalar],
    does_op_promote_integer_inputs_to_float: bool) -> bool {

    let does_op_promote_integer_inputs_to_float: bool = does_op_promote_integer_inputs_to_float.unwrap_or(false);

    todo!();
        /*
            #ifdef __HIP_PLATFORM_HCC__
      return false;
    #else
      return check_fast_path_restrictions(tensorLists, scalarList, does_op_promote_integer_inputs_to_float);
    #endif
        */
}

pub fn can_use_fast_route_b(
    tensors1:                                TensorList,
    tensors2:                                TensorList,
    does_op_promote_integer_inputs_to_float: bool) -> bool {

    let does_op_promote_integer_inputs_to_float: bool = does_op_promote_integer_inputs_to_float.unwrap_or(false);

    todo!();
        /*
            #ifdef __HIP_PLATFORM_HCC__
      return false;
    #else
      return can_use_fast_route({tensors1, tensors2}, {}, does_op_promote_integer_inputs_to_float);
    #endif
        */
}
