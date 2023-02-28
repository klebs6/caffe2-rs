crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorCompare.h]

pub type ReduceMinmaxFn = fn(
        _0: &mut Tensor,
        _1: &mut Tensor,
        _2: &Tensor,
        _3: i64,
        _4: bool
) -> ();

declare_dispatch!{reduce_minmax_fn, max_stub}
declare_dispatch!{reduce_minmax_fn, min_stub}
declare_dispatch!{reduce_minmax_fn, _aminmax_stub}

pub type WhereFn = fn(_0: &mut TensorIterator, _1: ScalarType) -> ();

declare_dispatch!{
    where_fn, 
    where_kernel
}

pub type IsInfinityOpFn = fn(_0: &mut TensorIterator) -> ();

declare_dispatch!{is_infinity_op_fn, isposinf_stub}
declare_dispatch!{is_infinity_op_fn, isneginf_stub}

pub type ModeFn = fn(
        _0: &mut Tensor,
        _1: &mut Tensor,
        _2: &Tensor,
        _3: i64,
        _4: bool
) -> ();

declare_dispatch!{mode_fn, mode_stub}

pub type ClampFn = fn(_0: &mut TensorIterator) -> ();

declare_dispatch!{clamp_fn, clamp_stub}
declare_dispatch!{clamp_fn, clamp_min_stub}
declare_dispatch!{clamp_fn, clamp_max_stub}

declare_dispatch!{
    fn(
        _0: &mut TensorIterator,
        _1: Scalar,
        _2: Scalar
    ) -> c_void,
    clamp_scalar_stub
}

declare_dispatch!{
    fn(_0: &mut TensorIterator, _1: Scalar) -> c_void, 
    clamp_min_scalar_stub
}

declare_dispatch!{
    fn(_0: &mut TensorIterator, _1: Scalar) -> c_void,
    clamp_max_scalar_stub
}

pub type IsinDefaultFn = fn(
        _0: &Tensor,
        _1: &Tensor,
        _2: bool,
        _3: &Tensor
) -> ();

declare_dispatch!{isin_default_fn, isin_default_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorCompare.cpp]

#[inline] pub fn check_for_unsupported_isin_dtype(ty: ScalarType)  {
    
    todo!();
        /*
            // Bail out for dtypes unsupported by the sorting algorithm to keep the interface consistent.
      TORCH_CHECK(type != ScalarType::Bool &&
          type != ScalarType::BFloat16 &&
          type != ScalarType::ComplexFloat &&
          type != ScalarType::ComplexDouble,
          "Unsupported input type encountered for isin(): ", type);
        */
}

lazy_static!{
    /*
    TORCH_META_FUNC2(isin, Tensor_Tensor) (
      const Tensor& elements, const Tensor& test_elements, bool assume_unique, bool invert
    ) {
      check_for_unsupported_isin_dtype(elements.scalar_type());
      check_for_unsupported_isin_dtype(test_elements.scalar_type());
      set_output(elements.sizes(), TensorOptions(elements.device()).dtype(ScalarType::Bool));
    }

    TORCH_META_FUNC2(isin, Tensor_Scalar) (
      const Tensor& elements, const Scalar& test_elements, bool assume_unique, bool invert
    ) {
      check_for_unsupported_isin_dtype(elements.scalar_type());
      check_for_unsupported_isin_dtype(test_elements.type());
      set_output(elements.sizes(), TensorOptions(elements.device()).dtype(ScalarType::Bool));
    }

    TORCH_META_FUNC2(isin, Scalar_Tensor) (
      const Scalar& elements, const Tensor& test_elements, bool assume_unique, bool invert
    ) {
      check_for_unsupported_isin_dtype(elements.type());
      check_for_unsupported_isin_dtype(test_elements.scalar_type());
      set_output({0}, TensorOptions(test_elements.device()).dtype(ScalarType::Bool));
    }
    */
}

define_dispatch!{where_kernel}
define_dispatch!{max_stub}
define_dispatch!{min_stub}
define_dispatch!{_aminmax_stub}
define_dispatch!{isposinf_stub}
define_dispatch!{isneginf_stub}
define_dispatch!{mode_stub}
define_dispatch!{clamp_stub}
define_dispatch!{clamp_min_stub}
define_dispatch!{clamp_max_stub}
define_dispatch!{clamp_scalar_stub}
define_dispatch!{clamp_min_scalar_stub}
define_dispatch!{clamp_max_scalar_stub}
define_dispatch!{isin_default_stub}

pub fn allclose(
        self_:     &Tensor,
        other:     &Tensor,
        rtol:      f64,
        atol:      f64,
        equal_nan: bool) -> bool {
    
    todo!();
        /*
            return isclose(self, other, rtol, atol, equal_nan).all().item<u8>();
        */
}

/**
  | Note [closeness]
  |
  | A number A is close to B when either:
  |
  | (1) A is equal to B, with NaNs comparing equal
  | when equal_nan is true.
  |
  | (2) The error abs(A - B) is finite and less
  |      than the max error (atol + abs(rtol * B)).
  |
  | Note that this is consistent with NumPy's
  | isclose but divergent from Python's isclose,
  | which computes the max error symmetrically as
  | max(rtol * max(abs(A), abs(B)), atol).
  |
  | TODO: use bitwise operator overloads once we
  | add them
  |
  | TODO: revisit complex inputs and equal_nan=true
  |  after
  |  https://github.com/numpy/numpy/issues/15959 is
  |  resolved
  |
  */
pub fn isclose(
        self_:     &Tensor,
        other:     &Tensor,
        rtol:      f64,
        atol:      f64,
        equal_nan: bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.scalar_type() == other.scalar_type(), self.scalar_type(), " did not match ", other.scalar_type());
      TORCH_CHECK(!(self.is_complex() && equal_nan),
        "isclose with equal_nan=True is not supported for complex inputs.");
      TORCH_CHECK(!(self.is_quantized() || other.is_quantized()),
        "isclose is not supported for quantized inputs.");

      // Checks that rtol and atol are non-negative
      // Note: consistent with Python's isclose but divergent from NumPy's, which
      //  allows negative atol and rtol.
      TORCH_CHECK(rtol >= 0, "rtol must be greater than or equal to zero, but got ", rtol);
      TORCH_CHECK(atol >= 0, "atol must be greater than or equal to zero, but got ", atol);

      // Computes equality closeness
      Tensor close = self == other;
      if (equal_nan && self.is_floating_point()) {
          close.__ior__((self != self).__iand__(other != other));
      }

      // Note [closeness error computation]
      // atol and rtol are provided as doubles, so the computation
      // rtol * other will produce a float or complex tensor.
      // When the difference (self - other) is compared to it then the
      // tensor representing the difference will also be cast to float or complex.
      // However, since (self - other) in uint8 is very likely to produce a
      // negative value, this moves the cast forward so the difference is
      // always computed in a float or complex type.
      // If the values of the integer tensors cannot be exactly represented
      // by the default scalar type then this may cause an incorrect result.

      // Computes allowed and actual error
      Tensor cast_other;
      if (isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
        cast_other = other.to(get_default_dtype());
      } else {
        cast_other = other;
      }
      Tensor allowed_error = atol + (rtol * cast_other).abs();
      Tensor actual_error = (self - cast_other).abs();

      // Computes finite closeness
      close.__ior__(isfinite(actual_error).__iand__(actual_error <= allowed_error));

      return close;
        */
}


pub fn isnan(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self != self;
        */
}


pub fn isreal(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            // Note: Integral and Floating tensor values are always real
      if (isIntegralType(self.scalar_type(), /*includeBool=*/true) ||
          isFloatingType(self.scalar_type())) {
        return ones_like(self, kBool, MemoryFormat::Preserve);
      }

      return imag(self) == 0;
        */
}


pub fn isinf(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            // Note: Integral tensor values are never infinite
      if (isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
        return zeros_like(self, kBool, MemoryFormat::Preserve);
      }

      // Note: a complex value is infinite when either part is infinite
      if (self.is_complex()) {
        return isinf(real(self)).__ior__
              (isinf(imag(self)));
      }

      return AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(), "isinf", [&]() {
        return self.abs() == numeric_limits<Scalar>::infinity();
      });
        */
}


pub fn isposinf(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty_like(self, kBool, MemoryFormat::Preserve);
      isposinf_out(result, self);
      return result;
        */
}


pub fn isposinf_out(
        self_:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_complex(), "isposinf does not support complex inputs.");
      TORCH_CHECK(result.scalar_type() == kBool, "isposinf does not support non-boolean outputs.");
      result.resize_(self.sizes());

      if (isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
        result.fill_(false);
      } else {
        auto iter = TensorIteratorConfig()
          .check_all_same_dtype(false)
          .add_output(result)
          .add_input(self)
          .build();
        isposinf_stub(iter.device_type(), iter);
      }
      return result;
        */
}


pub fn isneginf(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty_like(self, kBool, MemoryFormat::Preserve);
      isneginf_out(result, self);
      return result;
        */
}


pub fn isneginf_out(
        self_:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_complex(), "isneginf does not support complex inputs.");
      TORCH_CHECK(result.scalar_type() == kBool, "isneginf does not support non-boolean outputs.");
      result.resize_(self.sizes());

      if (isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
        result.fill_(false);
      } else {
        auto iter = TensorIteratorConfig()
          .check_all_same_dtype(false)
          .add_output(result)
          .add_input(self)
          .build();
        isneginf_stub(iter.device_type(), iter);
      }
      return result;
        */
}


pub fn isfinite(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            // Note: Integral tensor values are always finite
      if (isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
        return ones_like(self, kBool, MemoryFormat::Preserve);
      }

      // Note: a complex value is finite iff both parts are finite
      if (self.is_complex()) {
        return isfinite(self.abs());
      }

      return AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "isfinite", [&]() {
        return (self == self) * (self.abs() != numeric_limits<Scalar>::infinity());
      });
        */
}


pub fn is_nonzero(self_: &Tensor) -> bool {
    
    todo!();
        /*
            auto n = self.numel();
      TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
      TORCH_CHECK(n < 2, "Boolean value of Tensor with more than one value is ambiguous");

      Scalar localScalar = self.item();
      if (localScalar.isFloatingPoint()) {
        return localScalar.to<double>() != 0;
      } else if (localScalar.isComplex()) {
         return localScalar.to<complex<double>>() != complex<double>(0.0, 0.0);
      } else if (localScalar.isIntegral(false)){
        return localScalar.to<i64>() != 0;
      } else if (localScalar.isBoolean()) {
        return localScalar.to<bool>();
      }
      TORCH_INTERNAL_ASSERT(false, "Expected non-Tensor backend scalar");
        */
}


pub fn assert_async_cpu(self_: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(native::is_nonzero(self), "Expected Tensor with single nonzero value, but got zero");
        */
}

/**
  | DO NOT USE THIS -- it's just an implementation
  | detail of wrapped_scalar tensor below.
  |
  */
pub fn scalar_to_tensor_default_dtype(
        s:      &Scalar,
        device: Device) -> Tensor {
    let device: Device = device.unwrap_or(kCPU);

    todo!();
        /*
            if (s.isFloatingPoint()) {
        return scalar_tensor(
            s, device(device).dtype(get_default_dtype()));
      } else if (s.isBoolean()) {
        return scalar_tensor(s, device(device).dtype(kBool));
      } else if (s.isComplex()) {
        return scalar_tensor(
            s, device(device).dtype(get_default_complex_dtype()));
      } else {
        TORCH_INTERNAL_ASSERT(s.isIntegral(false));
        return scalar_tensor(s, device(device).dtype(kLong));
      }
        */
}

/**
  | TLDR: Don't call
  | `wrapped_scalar_tensor_default_dtype` -- this
  | function is only necessary to support the
  | partial type-promotion that torch.where
  | supports.
  |
  | Once torch.where fully supports type promotion,
  | we won't need this function.
  |
  | Longer explanation:
  | `wrapped_scalar_tensor_default_dtype` is a bit
  | of a hack because torch.where doesn't support
  | type promotion, but does support
  | `torch.where(tensor, scalar1, scalar2)` with
  | default scalar types.
  |
  | The trickiness is we usually convert double
  | scalars to doubles, and `set_wrapped_number`
  | defines type promotion priority as being below
  | tensor types rather than as the default dtype
  | (perhaps we should?).
  |
  | This wouldn't matter if we just supported type
  | normal type promotion on torch.where, however.
  |
  */
pub fn wrapped_scalar_tensor_default_dtype(
    scalar: &Scalar,
    device: Device) -> Tensor {

    todo!();
        /*
            Tensor tensor;
      tensor = scalar_to_tensor_default_dtype(scalar, device);
      tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
      return tensor;
        */
}

/**
  | Sorting-based algorithm for isin();
  | used when the number of test elements
  | is large.
  |
  */
pub fn isin_sorting(
    elements:      &Tensor,
    test_elements: &Tensor,
    assume_unique: bool,
    invert:        bool,
    out:           &Tensor)  {

    todo!();
        /*
            // 1. Concatenate unique elements with unique test elements in 1D form. If
      //    assume_unique is true, skip calls to unique().
      Tensor elements_flat, test_elements_flat, unique_order;
      if (assume_unique) {
        elements_flat = elements.ravel();
        test_elements_flat = test_elements.ravel();
      } else {
        tie (elements_flat, unique_order) = _unique(
            elements, /*sorted=*/ false, /*return_inverse=*/ true);
        tie (test_elements_flat, ignore) = _unique(test_elements, /*sorted=*/ false);
      }

      // 2. Stable sort all elements, maintaining order indices to reverse the
      //    operation. Stable sort is necessary to keep elements before test
      //    elements within the sorted list.
      Tensor all_elements = _cat({elements_flat, test_elements_flat});
      Tensor sorted_elements, sorted_order;
      tie (sorted_elements, sorted_order) = all_elements.sort(
          /*stable=*/ true, /*dim=*/ 0, /*descending=*/ false);

      // 3. Create a mask for locations of adjacent duplicate values within the
      //    sorted list. Duplicate values are in both elements and test elements.
      Tensor duplicate_mask = empty_like(sorted_elements, TensorOptions(ScalarType::Bool));
      Tensor sorted_except_first = sorted_elements.slice(0, 1, indexing::None);
      Tensor sorted_except_last = sorted_elements.slice(0, 0, -1);
      duplicate_mask.slice(0, 0, -1).copy_(
        invert ? sorted_except_first.ne(sorted_except_last) : sorted_except_first.eq(sorted_except_last));
      duplicate_mask.index_put_({-1}, invert);

      // 4. Reorder the mask to match the pre-sorted element order.
      Tensor mask = empty_like(duplicate_mask);
      mask.index_copy_(0, sorted_order, duplicate_mask);

      // 5. Index the mask to match the pre-unique element order. If
      //    assume_unique is true, just take the first N items of the mask,
      //    where N is the original number of elements.
      if (assume_unique) {
        out.copy_(mask.slice(0, 0, elements.numel()).view_as(out));
      } else {
        out.copy_(index(mask, {optional<Tensor>(unique_order)}));
      }
        */
}

pub fn where_a(
        condition: &Tensor,
        self_:     &Tensor,
        other:     &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(condition.device() == self.device() && self.device() == other.device(),
                  "Expected condition, x and y to be on the same device, but condition is on ",
                  condition.device(), " and x and y are on ", self.device(), " and ", other.device(),
                  " respectively");

      if (condition.scalar_type() == ScalarType::Byte) {
      TORCH_WARN_ONCE("where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead.");
    } else {
      TORCH_CHECK(condition.scalar_type() == ScalarType::Bool, "where expected condition to be a boolean tensor, but got a tensor with dtype ", condition.scalar_type());
    }

      MaybeOwned<Tensor> b_condition, b_self, b_other;
      tie(b_condition, b_self, b_other) = expand_outplace(condition, self, other, "where");
      return _s_where(*b_condition, *b_self, *b_other);
        */
}

pub fn where_b(
        condition: &Tensor,
        self_:     &Scalar,
        other:     &Tensor) -> Tensor {
    
    todo!();
        /*
            return where(condition, wrapped_scalar_tensor(self, other.device()), other);
        */
}

pub fn where_c(
        condition: &Tensor,
        self_:     &Tensor,
        other:     &Scalar) -> Tensor {
    
    todo!();
        /*
            return where(condition, self, wrapped_scalar_tensor(other, self.device()));
        */
}


pub fn where_d(
        condition: &Tensor,
        self_:     &Scalar,
        other:     &Scalar) -> Tensor {
    
    todo!();
        /*
            const auto device = condition.device();
      const Tensor& other_t = wrapped_scalar_tensor_default_dtype(other, device);
      const Tensor& self_t = wrapped_scalar_tensor_default_dtype(self, device);
      return where(condition, self_t, other_t);
        */
}

pub fn where_e(condition: &Tensor) -> Vec<Tensor> {
    
    todo!();
        /*
            return condition.nonzero_numpy();
        */
}


pub fn s_where(
        condition: &Tensor,
        self_:     &Tensor,
        other:     &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.dtype() == other.dtype(), "expected scalar type ", self.dtype(), " but found ", other.dtype());
      Tensor ret = empty(self.sizes(), self.options());
      auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(ret)
        .add_input(condition)
        .add_input(self)
        .add_input(other)
        .build();
      where_kernel(iter.device_type(), iter, condition.scalar_type());
      return ret;
        */
}

pub fn mode_a(
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            Tensor values = empty({0}, self.options());
      Tensor indices = empty({0}, self.options().dtype(kLong));
      return native::mode_out(self, dim, keepdim, values, indices);
        */
}

pub fn mode_out_a(
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool,
        values:  &mut Tensor,
        indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
                  "mode only supports CPU AND CUDA device type, got: ", self.device().type());
      TORCH_CHECK(self.layout() == Layout::Strided,
                  "mode only supports strided layout, got: ", self.layout());
      TORCH_CHECK(self.device() == values.device(),
                  "expected device '", self.device(), "' but got '",
                  values.device(), "' for values output");
      TORCH_CHECK(self.device() == indices.device(),
                  "expected device '", self.device(), "' but got '",
                  indices.device(), "' for indices output");
      TORCH_CHECK(self.scalar_type() == values.scalar_type(),
                  "expected scalar type '", self.scalar_type(), "' but got '",
                  values.scalar_type(), "' for values output");
      TORCH_CHECK(indices.scalar_type() == ScalarType::Long,
                  "expected scalar type '", ScalarType::Long, "' but got '",
                  indices.scalar_type(), "' for indices output");
      dim = maybe_wrap_dim(dim, self.dim());
      if (self.numel() == 0) {
        zero_numel_tensor_resize(values, indices, self, dim, keepdim, "mode()");
        return tie(values, indices);
      }
      else if (_dimreduce_return_trivial_no_ident(values, self, dim, keepdim, "mode")) {
        AT_ASSERT(values.dim() == 0);
        indices.resize_({}).fill_(0);
        return forward_as_tuple(values, indices);
      } else {
        auto result = [&]() {
          NoNamesGuard guard;
          mode_stub(self.device().type(), values, indices, self, dim, keepdim);
          return tuple<Tensor &,Tensor &>{values, indices};
        }();
        namedinference::propagate_names_for_reduction(get<0>(result), self, dim, keepdim);
        namedinference::propagate_names_for_reduction(get<1>(result), self, dim, keepdim);
        return result;
      }
        */
}

pub fn max_b(
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            Tensor max_indices = empty({0}, self.options().dtype(kLong));
        if (self.is_quantized()) {
          Tensor max = empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
          native::max_out(self.int_repr(), dim, keepdim, max, max_indices);
          // TODO: qscheme
          return tuple<Tensor, Tensor>(_make_per_tensor_quantized_tensor(max,
            self.q_scale(), self.q_zero_point()), max_indices);
        } else {
          Tensor max = empty({0}, self.options());
          return native::max_out(self, dim, keepdim, max, max_indices);
        }
        */
}

pub fn max_out_impl(
        max:         &mut Tensor,
        max_indices: &mut Tensor,
        self_:       &Tensor,
        dim:         i64,
        keepdim:     bool) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
                  "max only supports CPU AND CUDA device type, got: ", self.device().type());
      TORCH_CHECK(self.layout() == Layout::Strided,
                  "max only supports strided layout, got: ", self.layout());
      TORCH_CHECK(self.device() == max.device(),
                  "expected device ", self.device(), " but got ",
                  max.device(), " for max values output");
      TORCH_CHECK(self.device() == max_indices.device(),
                  "expected device ", self.device(), " but got ",
                  max_indices.device(), " for indices output");
      dim = maybe_wrap_dim(dim, self.dim());
      if (self.numel() == 0) {
        zero_numel_tensor_resize(max, max_indices, self, dim, keepdim, "max()");
        return tie(max, max_indices);
      }
      else if (_dimreduce_return_trivial_no_ident(max, self, dim, keepdim, "max")) {
        // case where self.numel() == 1. The result does not need to be reshaped
        // as a case of reduction in this case.
        TORCH_CHECK(!self.is_complex(), "max does not support complex inputs.");
        AT_ASSERT(max.dim() == 0);
        max_indices.resize_({}).fill_(0);
        return forward_as_tuple(max, max_indices);
      } else {
        max_stub(self.device().type(), max, max_indices, self, dim, keepdim);
        return tuple<Tensor &,Tensor &>{max, max_indices};
      }
        */
}

pub fn max_out_a(
        self_:       &Tensor,
        dim:         i64,
        keepdim:     bool,
        max:         &mut Tensor,
        max_indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        return max_out_impl(max, max_indices, self, dim, keepdim);
      }();
      namedinference::propagate_names_for_reduction(max, self, dim, keepdim);
      namedinference::propagate_names_for_reduction(max_indices, self, dim, keepdim);
      return result;
        */
}

pub fn min_a(
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            Tensor min_indices = empty({0}, self.options().dtype(kLong));
      if (self.is_quantized()) {
        Tensor min = empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
        native::min_out(self.int_repr(), dim, keepdim, min, min_indices);
        return tuple<Tensor, Tensor>(_make_per_tensor_quantized_tensor(min, self.q_scale(), self.q_zero_point()), min_indices);
      } else {
        Tensor min = empty({0}, self.options());
        return native::min_out(self, dim, keepdim, min, min_indices);
      }
        */
}

pub fn aminmax_out_impl(
        min:     &mut Tensor,
        max:     &mut Tensor,
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
                  "min_max_val only supports CPU AND CUDA device type, got: ", self.device().type());
      TORCH_CHECK(self.layout() == Layout::Strided,
                  "min_max only supports strided layout, got: ", self.layout());
      TORCH_CHECK(self.device() == min.device(),
                  "expected device ", self.device(), " but got ",
                  min.device(), " for min values output");
      TORCH_CHECK(self.device() == max.device(),
                  "expected device ", self.device(), " but got ",
                  max.device(), " for max values output");
      dim = maybe_wrap_dim(dim, self.dim());
      if (_dimreduce_return_trivial_no_ident(min, self, dim, keepdim, "min") &&
          _dimreduce_return_trivial_no_ident(max, self, dim, keepdim, "max")) {
        TORCH_CHECK(!self.is_complex(), "min_max does not support complex inputs.");
        return forward_as_tuple(min, max);
      } else {
        _aminmax_stub(self.device().type(), min, max, self, dim, keepdim);
        return tuple<Tensor &, Tensor &>{min, max};
      }
        */
}


pub fn aminmax(
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_quantized(), "min is not yet implemented for quantized tensors.");

      Tensor min = empty({0}, self.options());
      Tensor max = empty({0}, self.options());

      auto result = _aminmax_out_impl(min, max, self, dim, keepdim);
      return result;
        */
}


pub fn min_out_impl(
        min:         &mut Tensor,
        min_indices: &mut Tensor,
        self_:       &Tensor,
        dim:         i64,
        keepdim:     bool) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
                  "min only supports CPU AND CUDA device type, got: ", self.device().type());
      TORCH_CHECK(self.layout() == Layout::Strided,
                  "min only supports strided layout, got: ", self.layout());
      TORCH_CHECK(self.device() == min.device(),
                  "expected device ", self.device(), " but got ",
                  min.device(), " for min values output");
      TORCH_CHECK(self.device() == min_indices.device(),
                  "expected device ", self.device(), " but got ",
                  min_indices.device(), " for indices output");
      dim = maybe_wrap_dim(dim, self.dim());
      if (self.numel() == 0) {
        zero_numel_tensor_resize(min, min_indices, self, dim, keepdim, "min()");
        return tie(min, min_indices);
      }
      else if (_dimreduce_return_trivial_no_ident(min, self, dim, keepdim, "min")) {
        TORCH_CHECK(!self.is_complex(), "min does not support complex inputs.");
        AT_ASSERT(min.dim() == 0);
        min_indices.resize_({}).fill_(0);
        return forward_as_tuple(min, min_indices);
      } else {
        min_stub(self.device().type(), min, min_indices, self, dim, keepdim);
        return tuple<Tensor &,Tensor &>{min, min_indices};
      }
        */
}

pub fn min_out_b(
        self_:       &Tensor,
        dim:         i64,
        keepdim:     bool,
        min:         &mut Tensor,
        min_indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        return min_out_impl(min, min_indices, self, dim, keepdim);
      }();
      namedinference::propagate_names_for_reduction(min, self, dim, keepdim);
      namedinference::propagate_names_for_reduction(min_indices, self, dim, keepdim);
      return result;
        */
}


pub fn clamp_out_a(
    self_:  &Tensor,
    min:    &Option<Scalar>,
    max:    &Option<Scalar>,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            if (min && max) {
        auto iter = TensorIterator::unary_op(result, self);
        clamp_scalar_stub(iter.device_type(), iter, *min, *max);
      } else if (max) {
        clamp_max_outf(self, *max, result);
      } else if (min) {
        clamp_min_outf(self, *min, result);
      } else {
        TORCH_CHECK(false, "torch.clamp: At least one of 'min' or 'max' must not be None");
      }
      return result;
        */
}

pub fn clamp_out_b(
    self_:  &Tensor,
    min:    &Option<Tensor>,
    max:    &Option<Tensor>,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            if (min && max) {
        TORCH_CHECK(self.layout() == Layout::Strided,
                    "torch.clamp only supports strided layout, got: ", self.layout());
        auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(true)
                    .add_output(result)
                    .add_input(self)
                    .add_input(*min)
                    .add_input(*max)
                    .promote_inputs_to_common_dtype(true)
                    .cast_common_dtype_to_outputs(true)
                    .enforce_safe_casting_to_output(true)
                    .build();
        clamp_stub(iter.device_type(), iter);
      } else if (max) {
        clamp_max_outf(self, *max, result);
      } else if (min) {
        clamp_min_outf(self, *min, result);
      } else {
        TORCH_CHECK(false, "torch.clamp: At least one of 'min' or 'max' must not be None");
      }
      return result;
        */
}

pub fn clamp_a(
        self_: &Tensor,
        min:   &Option<Scalar>,
        max:   &Option<Scalar>) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return clamp_outf(self, min, max, result);
        */
}

pub fn clamp_b(
        self_: &Tensor,
        min:   &Option<Tensor>,
        max:   &Option<Tensor>) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return clamp_outf(self, min, max, result);
        */
}

pub fn clamp_c(
        self_: &mut Tensor,
        min:   &Option<Scalar>,
        max:   &Option<Scalar>) -> &mut Tensor {
    
    todo!();
        /*
            return clamp_outf(self, min, max, self);
        */
}

pub fn clamp_d(
        self_: &mut Tensor,
        min:   &Option<Tensor>,
        max:   &Option<Tensor>) -> &mut Tensor {
    
    todo!();
        /*
            return clamp_outf(self, min, max, self);
        */
}


pub fn clamp_max_out_a(
        self_:  &Tensor,
        max:    &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::unary_op(result, self);
      clamp_max_scalar_stub(iter.device_type(), iter, max);
      return result;
        */
}

pub fn clamp_max_out_b(
        self_:  &Tensor,
        max:    &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.layout() == Layout::Strided,
                  "torch.clamp only supports strided layout, got: ", self.layout());
      auto iter = TensorIterator::borrowing_binary_op(result, self, max);
      clamp_max_stub(iter.device_type(), iter);
      return result;
        */
}


pub fn clamp_max_a(
        self_: &Tensor,
        max:   &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return clamp_max_outf(self, max, result);
        */
}


pub fn clamp_max_b(
        self_: &Tensor,
        max:   &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return clamp_max_outf(self, max, result);
        */
}


pub fn clamp_max_c(
        self_: &mut Tensor,
        max:   &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return clamp_max_outf(self, max, self);
        */
}

pub fn clamp_max_d(
        self_: &mut Tensor,
        max:   &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return clamp_max_outf(self, max, self);
        */
}

pub fn clamp_min_out_a(
    self_:  &Tensor,
    min:    &Scalar,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto iter = TensorIterator::unary_op(result, self);
      clamp_min_scalar_stub(iter.device_type(), iter, min);
      return result;
        */
}

pub fn clamp_min_out_b(
    self_:  &Tensor,
    min:    &Tensor,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.layout() == Layout::Strided,
                  "torch.clamp only supports strided layout, got: ", self.layout());
      auto iter = TensorIterator::borrowing_binary_op(result, self, min);
      clamp_min_stub(iter.device_type(), iter);
      return result;
        */
}

pub fn clamp_min_a(
    self_: &Tensor,
    min:   &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return clamp_min_outf(self, min, result);
        */
}

pub fn clamp_min_b(
    self_: &Tensor,
    min:   &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return clamp_min_outf(self, min, result);
        */
}

pub fn clamp_min_c(
    self_: &mut Tensor,
    min:   &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return clamp_min_outf(self, min, self);
        */
}

pub fn clamp_min_d(
    self_: &mut Tensor,
    min:   &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return clamp_min_outf(self, min, self);
        */
}

/**
  | Implements the "clip" alias for clamp
  |
  */
pub fn clip_out_a(
    self_:  &Tensor,
    min:    &Option<Scalar>,
    max:    &Option<Scalar>,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return clamp_outf(self, min, max, result);
        */
}

pub fn clip_out_b(
    self_:  &Tensor,
    min:    &Option<Tensor>,
    max:    &Option<Tensor>,
    result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return clamp_outf(self, min, max, result);
        */
}


pub fn clip_a(
        self_: &Tensor,
        min:   &Option<Scalar>,
        max:   &Option<Scalar>) -> Tensor {
    
    todo!();
        /*
            return clamp(self, min, max);
        */
}

pub fn clip_b(
    self_: &Tensor,
    min:   &Option<Tensor>,
    max:   &Option<Tensor>) -> Tensor {
    
    todo!();
        /*
            return clamp(self, min, max);
        */
}

pub fn clip_c(
    self_: &mut Tensor,
    min:   &Option<Scalar>,
    max:   &Option<Scalar>) -> &mut Tensor {

    todo!();
        /*
            return clamp_(self, min, max);
        */
}

pub fn clip_d(
    self_: &mut Tensor,
    min:   &Option<Tensor>,
    max:   &Option<Tensor>) -> &mut Tensor {
    
    todo!();
        /*
            return clamp_(self, min, max);
        */
}

pub fn min_b(
    self_:   &Tensor,
    dim:     Dimname,
    keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return min(self, dimname_to_position(self, dim), keepdim);
        */
}

pub fn min_out_a(
    self_:       &Tensor,
    dim:         Dimname,
    keepdim:     bool,
    min:         &mut Tensor,
    min_indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {

    todo!();
        /*
            return min_out(min, min_indices, self, dimname_to_position(self, dim), keepdim);
        */
}


pub fn max_a(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return max(self, dimname_to_position(self, dim), keepdim);
        */
}

pub fn max_out_b(
        self_:       &Tensor,
        dim:         Dimname,
        keepdim:     bool,
        max:         &mut Tensor,
        max_indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            return max_out(max, max_indices, self, dimname_to_position(self, dim), keepdim);
        */
}


pub fn argmax(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("argmax");
        */
}


pub fn argmin(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("argmin");
        */
}

pub fn argsort(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("argsort");
        */
}

pub fn mode_b(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return mode(self, dimname_to_position(self, dim), keepdim);
        */
}

pub fn mode_out_b(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool,
        values:  &mut Tensor,
        indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            return mode_out(values, indices, self, dimname_to_position(self, dim), keepdim);
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(isin_Tensor_Tensor_out) (
      const Tensor& elements, const Tensor& test_elements, bool assume_unique, bool invert, const Tensor& out
    ) {
      if (elements.numel() == 0) {
        return;
      }

      // Heuristic taken from numpy's implementation.
      // See https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/arraysetops.py#L575
      if (test_elements.numel() < static_cast<i64>(
            10.0f * pow(static_cast<double>(elements.numel()), 0.145))) {
        out.fill_(invert);
        isin_default_stub(elements.device().type(), elements, test_elements, invert, out);
      } else {
        isin_sorting(elements, test_elements, assume_unique, invert, out);
      }
    }

    TORCH_IMPL_FUNC(isin_Tensor_Scalar_out) (
      const Tensor& elements, const Scalar& test_elements, bool assume_unique, bool invert, const Tensor& out
    ) {
      // redispatch to eq / ne
      if (invert) {
        ne_out(const_cast<Tensor&>(out), elements, test_elements);
      } else {
        eq_out(const_cast<Tensor&>(out), elements, test_elements);
      }
    }

    TORCH_IMPL_FUNC(isin_Scalar_Tensor_out) (
      const Scalar& elements, const Tensor& test_elements, bool assume_unique, bool invert, const Tensor& out
    ) {
      // redispatch
      isin_out(const_cast<Tensor&>(out), wrapped_scalar_tensor(elements, test_elements.device()),
        test_elements, assume_unique, invert);
    }
    */
}
