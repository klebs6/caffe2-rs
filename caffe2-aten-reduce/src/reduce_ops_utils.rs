crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ReduceOpsUtils.h]

/**
  | Maximum and minimum possible scalar
  | values, including infinities
  |
  */
pub fn upper_bound<Scalar>() -> Scalar {

    todo!();
        /*
            using lim = numeric_limits<Scalar>;
      return lim::has_infinity ? lim::infinity() : lim::max();
        */
}

pub fn lower_bound<Scalar>() -> Scalar {

    todo!();
        /*
            using lim = numeric_limits<Scalar>;
      return lim::has_infinity ? -lim::infinity() : lim::lowest();
        */
}

#[inline] pub fn ensure_nonempty_dim(dim: i64) -> i64 {
    
    todo!();
        /*
            return max<i64>(dim, 1);
        */
}


#[inline] pub fn ensure_nonempty_size(
        t:   &Tensor,
        dim: i64) -> i64 {
    
    todo!();
        /*
            return t.dim() == 0 ? 1 : t.size(dim);
        */
}


#[inline] pub fn ensure_nonempty_stride(
        t:   &Tensor,
        dim: i64) -> i64 {
    
    todo!();
        /*
            return t.dim() == 0 ? 1 : t.stride(dim);
        */
}

pub type IdxVec = Vec<i64>;


#[inline] pub fn ensure_nonempty_vec(vec: IdxVec) -> IdxVec {
    
    todo!();
        /*
            if (vec.size() == 0) {
        vec.push_back(1);
      }
      return vec;
        */
}


#[inline] pub fn restride_dim(
        src:               &Tensor,
        dim:               i64,
        replacement_shape: &[i32]) -> Tensor {
    
    todo!();
        /*
            auto strides = ensure_nonempty_vec(src.strides().vec());
      strides[dim] = 0;
      return src.as_strided(replacement_shape, strides);
        */
}


#[inline] pub fn dimreduce_setup<'a>(
        result: &mut Tensor,
        self_:  &Tensor,
        dim:    i64) -> &'a mut Tensor {
    
    todo!();
        /*
            IntArrayRef self_sizes = self.sizes();
      vector<i64> result_sizes;
      result_sizes.insert(result_sizes.end(), self_sizes.begin(), self_sizes.end());
      result_sizes[dim] = 1;
      result.resize_(result_sizes);
      return result;
        */
}


#[inline] pub fn dimreduce_return_trivial(
        result:  &mut Tensor,
        self_:   &Tensor,
        ident:   &Scalar,
        dim:     i64,
        keepdim: bool) -> bool {
    
    todo!();
        /*
            if (self.numel() == 1 && self.ndimension() == 0) {
        result.resize_({});
        result.fill_(self);
        return true;
      }
      // Return identity
      if (self.numel() == 0) {
        _dimreduce_setup(result, self, dim);
        result.fill_(ident);
        if (!keepdim) result.squeeze_(dim);
        return true;
      }
      return false;
        */
}


#[inline] pub fn dimreduce_return_trivial_no_ident(
        result:  &mut Tensor,
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool,
        fn_name: *const u8) -> bool {
    
    todo!();
        /*
            if (self.numel() == 1 && self.ndimension() == 0) {
        result.resize_({});
        result.fill_(self);
        return true;
      }

      return false;
        */
}


#[inline] pub fn allreduce_return_trivial(
        self_: &Tensor,
        ident: &Scalar) -> Option<Tensor> {
    
    todo!();
        /*
            // Return identity
      if (self.numel() == 0) {
        return scalar_tensor(ident, self.options());
      }
      return nullopt;
        */
}


#[macro_export] macro_rules! option_type_equality_check {
    ($option:ident, $out:ident, $self:ident) => {
        /*
        
        { 
          TORCH_CHECK(
            out.option() == self.option(),
            "expected ", #option, " ",
            self.option(),
            " but found ", out.option())
        }
        */
    }
}


#[inline] pub fn check_scalar_type_device_layout_equal(
        out:   &Tensor,
        self_: &Tensor)  {
    
    todo!();
        /*
            OPTION_TYPE_EQUALITY_CHECK(scalar_type, out, self);
      OPTION_TYPE_EQUALITY_CHECK(device, out.options(), self.options());
      OPTION_TYPE_EQUALITY_CHECK(layout, out.options(), self.options());
        */
}


#[inline] pub fn integer_upcast(
        self_: &Tensor,
        dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            ScalarType scalarType = self.scalar_type();
      ScalarType upcast_scalarType = dtype.value_or(isIntegralType(scalarType, /*includeBool=*/true) ? ScalarType::Long : scalarType);
      return self.toType(upcast_scalarType);
        */
}

pub type DimMask = TensorIterator_DimMask;


pub fn make_dim_mask(
        dims: &[i32],
        ndim: i64) -> DimMask {
    
    todo!();
        /*
            DimMask mask;
      if (dims.empty()) {
        mask = DimMask().flip();
      } else {
        mask = dim_list_to_bitset(dims, ndim);
      }
      return mask;
        */
}


#[inline] pub fn shape_from_dim_mask(
        self_:   &Tensor,
        mask:    DimMask,
        keepdim: bool) -> DimVector {
    
    todo!();
        /*
            auto shape = DimVector(self.sizes());
      for (int dim = shape.size() - 1; dim >= 0; dim--) {
        if (mask[dim]) {
          if (keepdim) {
            shape[dim] = 1;
          } else {
            shape.erase(shape.begin() + dim);
          }
        }
      }
      return shape;
        */
}


pub fn resize_reduction_result(
        result:  &mut Tensor,
        self_:   &Tensor,
        mask:    DimMask,
        keepdim: bool,
        dtype:   ScalarType)  {
    
    todo!();
        /*
            auto shape = shape_from_dim_mask(self, mask, keepdim);
      TORCH_CHECK(result.defined(), "Cannot create a new tensor inside a reduction op. You likely tried to call an operator with an out argument but the out argument was an undefined tensor.");
      native::resize_output(result, shape);
        */
}

#[inline] pub fn create_reduction_result(
    self_:   &Tensor,
    dim:     &[i32],
    keepdim: bool,
    dtype:   ScalarType) -> Tensor {
    
    todo!();
        /*
            DimMask mask = make_dim_mask(dim, self.dim());
      auto shape = shape_from_dim_mask(self, mask, keepdim);
      return empty(shape, self.options().dtype(dtype));
        */
}

pub fn review_reduce_result(
    result:  &Tensor,
    ndim:    i32,
    mask:    DimMask,
    keepdim: bool) -> Tensor {
    
    todo!();
        /*
            if (keepdim) {
        return result;
      }
      auto shape = DimVector(result.sizes());
      auto stride = DimVector(result.strides());
      for (int dim = 0; dim < ndim; dim++) {
        if (mask[dim]) {
          shape.insert(shape.begin() + dim, 1);
          stride.insert(stride.begin() + dim, 0);
        }
      }
      return result.as_strided(shape, stride);
        */
}


pub fn make_reduction_a(
    name:      *const u8,
    result:    &mut Tensor,
    self_:     &Tensor,
    dim_opt:   Option<&[i32]>,
    keepdim:   bool,
    in_dtype:  ScalarType,
    out_dtype: ScalarType) -> TensorIterator {

    todo!();
    /*
            // check that result type and dtype match if provided
      TORCH_CHECK(
          !result.defined() || result.scalar_type() == out_dtype,
          name, ": provided dtype must match dtype of result. Got ",
          toString(result.scalar_type()),
          " and ",
          toString(out_dtype),
          ".");
      // dim={} performs an all-reduce, same as dim=None
      IntArrayRef dim = dim_opt.value_or(IntArrayRef{});
      i64 ndim = self.dim();
      auto mask = make_dim_mask(dim, ndim);
      resize_reduction_result(result, self, mask, keepdim, out_dtype);
      auto viewed_result = review_reduce_result(result, ndim, mask, keepdim);
      namedinference::propagate_names_for_reduction(result, self, dim, keepdim);
      if (self.scalar_type() == in_dtype) {
        return TensorIterator::reduce_op(viewed_result, self);
      }
      return TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
        */
}

pub fn make_reduction_b(
    name:      *const u8,
    result:    &mut Tensor,
    self_:     &Tensor,
    dim:       Option<&[i32]>,
    keepdim:   bool,
    out_dtype: ScalarType) -> TensorIterator {

    todo!();
    /*
            // special case for type promotion in mixed precision, improves computational
      // efficiency.
      // not generalize this to common mismatched input/output types to avoid cross
      // product of templated kernel launches.
      const bool gpu_lowp_to_f32 = (
        self.is_cuda() && (self.scalar_type() == kHalf || self.scalar_type() == kBFloat16) && out_dtype == kFloat);
      auto in_dtype = gpu_lowp_to_f32 ? self.scalar_type() : out_dtype;
      return make_reduction(name, result, self, dim, keepdim, in_dtype, out_dtype);
        */
}


pub fn make_reduction_c(
    name:    *const u8,
    result1: &mut Tensor,
    result2: &mut Tensor,
    self_:   &Tensor,
    dim_opt: Option<&[i32]>,
    keepdim: bool,
    dtype1:  ScalarType,
    dtype2:  ScalarType) -> TensorIterator {
    
    todo!();
        /*
            // check that result type and dtype match if provided
      TORCH_CHECK(
        (!result1.defined() || result1.scalar_type() == dtype1) && (!result2.defined() || result2.scalar_type() == dtype2),
        name, ": provided dtype must match dtype of result. Got ",
        toString(result1.scalar_type()), toString(result2.scalar_type()),
        " and ",
        toString(dtype1), toString(dtype2),
        ".");

      // dim={} performs an all-reduce, same as dim=None
      auto dim = dim_opt.value_or(IntArrayRef{});
      i64 ndim = self.dim();
      DimMask mask = make_dim_mask(dim, ndim);
      resize_reduction_result(result1, self, mask, keepdim, dtype1);
      auto viewed_result1 = review_reduce_result(result1, ndim, mask, keepdim);

      resize_reduction_result(result2, self, mask, keepdim, dtype2);
      auto viewed_result2 = review_reduce_result(result2, ndim, mask, keepdim);

      namedinference::propagate_names_for_reduction(result1, self, dim, keepdim);
      namedinference::propagate_names_for_reduction(result2, self, dim, keepdim);

      // special case for type promotion in mixed precision, improves computational
      // efficiency.
      // We don't generalize this to common mismatched input/output types to avoid cross
      // product of templated kernel launches.
      if (self.scalar_type() == dtype1 ||
          (self.is_cuda() && self.scalar_type() == kHalf && dtype1 == kFloat)) {
        return TensorIterator::reduce_op(viewed_result1, viewed_result2, self);
      }
      return TensorIterator::reduce_op(viewed_result1, viewed_result2, self.to(dtype1));
        */
}

pub fn make_reduction_d(
    name:    *const u8,
    result1: &mut Tensor,
    result2: &mut Tensor,
    self_:   &Tensor,
    dim:     Option<&[i32]>,
    keepdim: bool,
    dtype:   ScalarType) -> TensorIterator {
    
    todo!();
        /*
            return make_reduction(name, result1, result2, self, dim, keepdim, dtype, dtype);
        */
}

pub fn zero_numel_check_dims_a(
    self_:   &Tensor,
    dim:     i64,
    fn_name: *const u8)  {
    
    todo!();
        /*
            if (self.ndimension() == 0) {
        TORCH_CHECK_INDEX(dim == 0 || dim == -1, fn_name,
          ": Expected reduction dim -1 or 0 for scalar but got ", dim);
      }
      else {
        TORCH_CHECK_INDEX(self.size(dim) != 0, fn_name,
          ": Expected reduction dim ", dim, " to have non-zero size.");
      }
        */
}

pub fn zero_numel_check_dims_b(
    self_:   &Tensor,
    dim:     &[i32],
    fn_name: *const u8)  {
    
    todo!();
        /*
            for (const i64 d : dim) {
        zero_numel_check_dims(self, d, fn_name);
      }
        */
}

/**
  | Resize the result tensor and indices when
  | result.numel() == 0 depending on values of dim
  | and keepdim for returning tensors containing
  | reduction results.
  |
  | This function should be called when you are
  | reducing a zero-numel tensor and want to resize
  | the output and return it. This function exists
  | for resizing zero-numel tensors when the size
  | of the reduction dimension is non-zero.
  |
  */
pub fn zero_numel_tensor_resize(
    result:         &mut Tensor,
    result_indices: &mut Tensor,
    self_:          &Tensor,
    dim:            i64,
    keepdim:        bool,
    fn_name:        *const u8)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(self.numel() == 0,  fn_name, ": Expected self.numel() != 0.");
      zero_numel_check_dims(self, dim, fn_name);
      vector<i64> sizes;
      if (keepdim) {
        sizes = self.sizes().vec();
        sizes[dim] = 1;
      }
      else {
        for (const auto d : irange(self.dim())) {
          if (d != dim) {
            sizes.push_back(self.sizes()[d]);
          }
        }
      }
      native::resize_output(result, sizes);
      native::resize_output(result_indices, sizes);
        */
}
