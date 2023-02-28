crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/accumulate.h]

/**
  | Sum of a list of integers; accumulates
  | into the int64_t datatype
  |
  */
#[inline] pub fn sum_integers_a<X: PrimInt>(container: &[X]) -> i64 {
    
    todo!();
        /*
            // accumulate infers return type from `init` type, so if the `init` type
      // is not large enough to hold the result, computation can overflow. We use
      // `int64_t` here to avoid this.
      return accumulate(
          container.begin(), container.end(), static_cast<int64_t>(0));
        */
}

/**
  | Product of a list of integers; accumulates
  | into the int64_t datatype
  |
  */
#[inline] pub fn multiply_integers_a<X: PrimInt>(container: &[X]) -> i64 {
    
    todo!();
        /*
            // accumulate infers return type from `init` type, so if the `init` type
      // is not large enough to hold the result, computation can overflow. We use
      // `int64_t` here to avoid this.
      return accumulate(
          container.begin(),
          container.end(),
          static_cast<int64_t>(1),
          multiplies<int64_t>());
        */
}


/**
  | Return product of all dimensions starting from k
  |
  | Returns 1 if k>=dims.size()
  |
  */
#[inline] pub fn numelements_from_dim<C: PrimInt>(k: i32, dims: &C) -> i64 {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(k >= 0);

      if (k > dims.size()) {
        return 1;
      } else {
        auto cbegin = dims.cbegin();
        advance(cbegin, k);
        return multiply_integers(cbegin, dims.cend());
      }
        */
}

/**
  | Product of all dims up to k (not including
  | dims[k]) Throws an error if
  | k>dims.size()
  |
  */
#[inline] pub fn numelements_to_dim<C: PrimInt>(k: i32, dims: &C) -> i64 {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(0 <= k);
      TORCH_INTERNAL_ASSERT((unsigned)k <= dims.size());

      auto cend = dims.cbegin();
      advance(cend, k);
      return multiply_integers(dims.cbegin(), cend);
        */
}

/**
  | Product of all dims between k and l (including
  | dims[k] and excluding dims[l]) k and
  | l may be supplied in either order
  |
  */
#[inline] pub fn numelements_between_dim<C: PrimInt>(
        k:    i32,
        l:    i32,
        dims: &C) -> i64 {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(0 <= k);
      TORCH_INTERNAL_ASSERT(0 <= l);

      if (k > l) {
        swap(k, l);
      }

      TORCH_INTERNAL_ASSERT((unsigned)l < dims.size());

      auto cbegin = dims.cbegin();
      auto cend = dims.cbegin();
      advance(cbegin, k);
      advance(cend, l);
      return multiply_integers(cbegin, cend);
        */
}
