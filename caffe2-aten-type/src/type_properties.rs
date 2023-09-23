crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TypeProperties.h]

pub struct ResultTypeState {

    /**
      | = ScalarType::Undefined;
      |
      */
    dim_result:     ScalarType,


    /**
      | = ScalarType::Undefined;
      |
      */
    wrapped_result: ScalarType,


    /**
      | = ScalarType::Undefined;
      |
      */
    zero_result:    ScalarType,
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TypeProperties.cpp]

pub fn is_cuda(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return self.is_cuda();
        */
}


pub fn is_distributed(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return false;
        */
}


pub fn is_complex(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return self.is_complex();
        */
}


pub fn is_floating_point(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return self.is_floating_point();
        */
}


pub fn is_inference(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return self.is_inference();
        */
}


pub fn is_signed(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return self.is_signed();
        */
}


pub fn is_conj(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return self.is_conj();
        */
}


pub fn is_sparse(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return self.is_sparse();
        */
}


pub fn is_sparse_csr(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return self.is_sparse_csr();
        */
}


pub fn is_quantized(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return self.is_quantized();
        */
}

/**
  | True if `self` and `from` have compatible
  | tensor type so that `from`'s TensorImpl
  | can be copied to `self`.
  |
  */
pub fn has_compatible_shallow_copy_type(
        self_: &Tensor,
        from:  &Tensor) -> bool {
    
    todo!();
        /*
            return self.unsafeGetTensorImpl()->has_compatible_shallow_copy_type(
          from.key_set());
        */
}


pub fn type_as(
        self_: &Tensor,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.to(other.options());
        */
}


#[inline] pub fn promote_skip_undefined(
        a: ScalarType,
        b: ScalarType) -> ScalarType {
    
    todo!();
        /*
            if (a == ScalarType::Undefined) {
        return b;
      }
      if (b == ScalarType::Undefined) {
        return a;
      }
      return promoteTypes(a, b);
        */
}


#[inline] pub fn combine_categories(
        higher: ScalarType,
        lower:  ScalarType) -> ScalarType {
    
    todo!();
        /*
      if(isComplexType(higher)) {
        return higher;
      }
      else if(!isComplexType(lower) && isFloatingType(higher)) {
        return higher;
      }
      if (higher == ScalarType::Bool || isFloatingType(lower) || isComplexType(lower)) {
        return promote_skip_undefined(higher, lower);
      }
      if (higher != ScalarType::Undefined) {
          return higher;
      }
      return lower;
        */
}


pub fn update_result_type_state_a(
        tensor:   &Tensor,
        in_state: &ResultTypeState) -> ResultTypeState {
    
    todo!();
        /*
            if (!tensor.defined()) {
        return in_state;
      }
      ResultTypeState new_state = in_state;
      ScalarType current = tensor.scalar_type();
      if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        if(isComplexType(current)) {
          current = typeMetaToScalarType(get_default_complex_dtype());
        }
        else if(isFloatingType(current)) {
          current = typeMetaToScalarType(get_default_dtype());
        }
      }
      if ( tensor.dim() > 0 ) {
        new_state.dimResult = promote_skip_undefined(in_state.dimResult, current);
      } else if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
      } else {
        new_state.zeroResult = promote_skip_undefined(in_state.zeroResult, current);
      }
      return new_state;
        */
}

pub fn update_result_type_state_b(
        scalar:   &Scalar,
        in_state: &ResultTypeState) -> ResultTypeState {
    
    todo!();
        /*
            ResultTypeState new_state = in_state;
      ScalarType current = scalar.type();
      if (isComplexType(current)) {
        current = typeMetaToScalarType(get_default_complex_dtype());
      } else if (isFloatingType(current)) {
        current = typeMetaToScalarType(get_default_dtype());
      }
      new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
      return new_state;
        */
}


pub fn result_type_a(in_state: &ResultTypeState) -> ScalarType {
    
    todo!();
        /*
            return combine_categories(in_state.dimResult, combine_categories(in_state.zeroResult, in_state.wrappedResult));
        */
}


pub fn result_type_b(tensors: &[Tensor]) -> ScalarType {
    
    todo!();
        /*
            ResultTypeState state = {};
      for (const Tensor& tensor : tensors) {
        state = update_result_type_state(tensor, state);
      }
      return result_type(state);
        */
}


pub fn result_type_c(
        tensor: &Tensor,
        other:  &Tensor) -> ScalarType {
    
    todo!();
        /*
      vector<Tensor> tensors{move(tensor), move(other)};
      return native::result_type(tensors);
        */
}


pub fn result_type_d(
        tensor: &Tensor,
        other:  &Scalar) -> ScalarType {
    
    todo!();
        /*
            ResultTypeState state = {};
      state = update_result_type_state(tensor, state);
      state = update_result_type_state(other, state);
      return result_type(state);
        */
}


pub fn result_type_e(
        scalar: &Scalar,
        tensor: &Tensor) -> ScalarType {
    
    todo!();
        /*
            return result_type(tensor, scalar);
        */
}


pub fn result_type_f(
        scalar1: &Scalar,
        scalar2: &Scalar) -> ScalarType {
    
    todo!();
        /*
            ResultTypeState state = {};
      state = update_result_type_state(scalar1, state);
      state = update_result_type_state(scalar2, state);
      return result_type(state);
        */
}


pub fn can_cast(
        from: ScalarType,
        to:   ScalarType) -> bool {
    
    todo!();
        /*
            return canCast(from, to);
        */
}


pub fn promote_types(
        type1: ScalarType,
        type2: ScalarType) -> ScalarType {
    
    todo!();
        /*
            ScalarType ret = promoteTypes(type1, type2);
      TORCH_CHECK(ret != ScalarType::Undefined, "Promotion from ", type1, " and ", type2, " is unsupported.");
      return ret;
        */
}
