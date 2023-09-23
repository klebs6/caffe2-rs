crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ReduceOps.h]

pub type ReduceFn = fn(_0: &mut TensorIterator) -> ();

declare_dispatch!{reduce_fn, sum_stub}
declare_dispatch!{reduce_fn, nansum_stub}
declare_dispatch!{reduce_fn, prod_stub}
declare_dispatch!{reduce_fn, mean_stub}
declare_dispatch!{reduce_fn, and_stub}
declare_dispatch!{reduce_fn, or_stub}
declare_dispatch!{reduce_fn, min_values_stub}
declare_dispatch!{reduce_fn, max_values_stub}
declare_dispatch!{reduce_fn, argmax_stub}
declare_dispatch!{reduce_fn, argmin_stub}

pub type ReduceStdVarFunction = fn(
        _0:         &mut TensorIterator,
        correction: i64,
        take_sqrt:  bool
) -> ();

declare_dispatch!{reduce_std_var_function, std_var_stub}

pub type ReduceNormFn = fn(
        _0: &mut Tensor,
        _1: &Tensor,
        _2: &Scalar,
        _3: Option<i64>
) -> ();

declare_dispatch!{reduce_norm_fn, norm_kernel}

pub type ReduceFnFlag = fn(_0: &mut TensorIterator, _1: &Scalar) -> ();


declare_dispatch!{reduce_fn_flag, norm_stub}

pub type CumFn = fn(
        _0: &mut Tensor,
        _1: &Tensor,
        _2: i64
) -> ();


declare_dispatch!{cum_fn, cumsum_stub}
declare_dispatch!{cum_fn, cumprod_stub}
declare_dispatch!{cum_fn, logcumsumexp_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ReduceOps.cpp]

define_dispatch!{sum_stub}
define_dispatch!{nansum_stub}
define_dispatch!{std_var_stub}
define_dispatch!{prod_stub}
define_dispatch!{norm_stub}
define_dispatch!{mean_stub}
define_dispatch!{and_stub}
define_dispatch!{or_stub}
define_dispatch!{min_values_stub}
define_dispatch!{max_values_stub}
define_dispatch!{argmax_stub}
define_dispatch!{argmin_stub}
define_dispatch!{cumsum_stub}
define_dispatch!{cumprod_stub}
define_dispatch!{logcumsumexp_stub}

pub fn logcumsumexp_cpu(
        self_: &Tensor,
        dim:   i64) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty_like(self, MemoryFormat::Contiguous);
      return _logcumsumexp_out_cpu(self, dim, result);
        */
}

pub fn logcumsumexp_out_cpu<'a>(
        self_:  &Tensor,
        dim:    i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            logcumsumexp_stub(self.device().type(), result, self, dim);
      return result;
        */
}

pub fn logcumsumexp(
        self_: &Tensor,
        dim:   i64) -> Tensor {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        return _logcumsumexp(self, dim);
      }();
      namedinference::propagate_names(result, self);
      return result;
        */
}

pub fn cumsum_cpu(
        self_: &Tensor,
        dim:   i64) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty_like(self, MemoryFormat::Contiguous);
      cumsum_stub(self.device().type(), result, self, dim);
      return result;
        */
}


pub fn cumsum_out_cpu<'a>(
        self_:  &Tensor,
        dim:    i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            cumsum_stub(self.device().type(), result, self, dim);
      return result;
        */
}

pub fn cumsum_a(
        self_: &Tensor,
        dim:   Dimname,
        dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            return cumsum(self, dimname_to_position(self, dim), dtype);
        */
}


pub fn cumsum_b(
        self_: &Tensor,
        dim:   i64,
        dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        return _cumsum(integer_upcast(self, dtype), dim);
      }();
      namedinference::propagate_names(result, self);
      return result;
        */
}

pub fn cumsum_c<'a>(
        self_: &mut Tensor,
        dim:   i64,
        dtype: Option<ScalarType>) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
              !dtype.has_value() || (self.scalar_type() == dtype.value()),
              "provided dtype must match the dtype of self tensor in cumsum. Got ",
              toString(self.scalar_type()),
              " and ",
              toString(dtype.value()),
              ".");

      return _cumsum_out(self, self, dim);
        */
}

pub fn cumsum_out_a<'a>(
        self_:  &Tensor,
        dim:    Dimname,
        dtype:  Option<ScalarType>,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return cumsum_out(result, self, dimname_to_position(self, dim), dtype);
        */
}

pub fn cumsum_out_b<'a>(
        self_:  &Tensor,
        dim:    i64,
        dtype:  Option<ScalarType>,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
      TORCH_CHECK(
          !dtype.has_value() || (result.scalar_type() == dtype.value()),
          "provided dtype must match dtype of result in cumsum. Got ",
          toString(result.scalar_type()),
          " and ",
          toString(dtype.value()),
          ".");
      {
        NoNamesGuard guard;
        _cumsum_out(result, self.toType(result.scalar_type()), dim);
      }
      namedinference::propagate_names(result, self);
      return result;
        */
}

pub fn cumprod_cpu(
        self_: &Tensor,
        dim:   i64) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty_like(self, MemoryFormat::Contiguous);
      cumprod_stub(self.device().type(), result, self, dim);
      return result;
        */
}

pub fn cumprod_out_cpu<'a>(
        self_:  &Tensor,
        dim:    i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            cumprod_stub(self.device().type(), result, self, dim);
      return result;
        */
}

pub fn cumprod_a(
        self_: &Tensor,
        dim:   Dimname,
        dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            return cumprod(self, dimname_to_position(self, dim), dtype);
        */
}


pub fn cumprod_b<'a>(
        self_: &mut Tensor,
        dim:   Dimname,
        dtype: Option<ScalarType>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::cumprod_(self, dimname_to_position(self, dim), dtype);
        */
}


pub fn cumprod_c(
        self_: &Tensor,
        dim:   i64,
        dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        return _cumprod(integer_upcast(self, dtype), dim);
      }();
      namedinference::propagate_names(result, self);
      return result;
        */
}

pub fn cumprod_d<'a>(
        self_: &mut Tensor,
        dim:   i64,
        dtype: Option<ScalarType>) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
                !dtype.has_value() || (self.scalar_type() == dtype.value()),
                "provided dtype must match the dtype of self tensor in cumprod. Got ",
                toString(self.scalar_type()),
                " and ",
                toString(dtype.value()),
                ".");

        return _cumprod_out(self, self, dim);
        */
}


pub fn reversed_cumsum(
    w:   &Tensor,
    dim: i64) -> Tensor {
    
    todo!();
        /*
            /* Logically implements w.flip(dim).cumsum(dim).flip(dim) without copying. */
      const auto w_cumsum = w.cumsum(dim);
      const auto w_sum = w_cumsum.narrow(dim, -1, 1);
      return w_sum - w_cumsum + w;
        */
}

/**
  | We show here how to derive an O(n) gradient formula for
  | abitrary inputs. It follows via a basic application of the
  | chain rule together with a number of observations for different
  | cases. We assume that x is an n-dimensional vector and y = cumprod(x).
  | In the actual implementation we will need to play a bit with masks
  | to be able to implement the formulas deduced here for tensors.
  |
  | We will first deduce the formula for the case when
  | x[i] != 0 for 1 <= i <= n.
  |
  | For F : R^n -> R the cost function (we will look at the complex case later),
  | we have
  |
  | dF / dx_k = sum_j (dF / dy_j) * (dy_j / dx_k)   (1)
  |
  | The term dF / dy_j is just grad_output[j] (assuming again
  | everything is one-dimensional).
  |
  | The term (dy_j / dx_k) is easilly seen to be
  |
  | if j >= k
  |   dy_j / dx_k = prod_{1 <= i <= j, i != k} x_i
  | else:
  |   dy_j / dx_k = 0
  |
  | Note that the indicator (j>=k) can be taken out
  | by replacing the sum in (1) with a sum from
  | k <= j <= n.
  |
  | Thus,
  | dF / dx_k = sum_{k <= j <= n} grad_output[j] * (dy_j / dx_k)
  |
  | with
  | dy_j / dx_k = prod_{1 <= i <= j, i != k} x_i     (2)
  |
  | Note that this last term is just the cumulative product
  | with k omitted. Thus, if x_k (the input) is nonzero, we can
  | just express this as
  |
  | dy_j / dx_k = (prod_{1 <= i <= j} x_i) / x_k
  |             = y_j / x_k
  |
  | So therefore,
  |
  | dF / dx_k = sum_{k <= j <= n} grad_output[j] * y_j / x_k
  |
  | This formula just makes sense when input[i] != 0 for every i.
  |
  | Assume now that there exists at least a zero in the input.
  | Denote by z1 the first element 1 <= z1 <= n with input[z1] = 0
  | and z2 the second element z1 < z2 <= n with input[z2] = 0,
  | (or z2 = n if there is just one zero in input)
  |
  | We have three cases.
  |
  | k > z1:
  | Looking at (2), we see that dy_j / dx_k = 0, for j >= k, as these terms
  | all include a x_{z1} which is zero. As such, dF / dx_k = 0 in this case
  |
  | k < z1:
  | Reasoning as in the previous case, we see that for these elements we have that
  |
  | dF / dx_k = sum_{k <= j < z1} grad_output[j] * (dy_j / dx_k)
  |
  | as the terms of the sum for j in z1 <= j <= n are all zero
  |
  | k = z1:
  | Similar to the case k < z1, we have that
  |
  | dF / dx_z1 = sum_{z1 <= j < z2} grad_output[j] * (dy_j / dx_z1)
  |
  | This case has a subtlety though. To compute (dy_j / dx_z1), we cannot use the formula
  |
  | dy_j / dx_z1 = y_j / x_z1
  |
  | as, y_j = x_z1 = 0 for j >= z1. We need to compute it with the formula for its derivative,
  | that is:
  |
  | dy_j / dx_z1 = prod(x[:z1]) * (grad_output[z1] + sum(grad_output[z1+1:z2] * cumprod(x[z1+1:z2])))
  |
  | When the imputs are complex, this is map is holomorphic. As such, to compute
  | its backwards is just the conjugate of the usual backwards. This simplifies to
  | conjugating the input. We may also reuse the output as, since the map is holomorphic,
  | cumprod(input.conj()) = cumprod(input).conj()
  */
pub fn cumprod_backward(
        grad:   &Tensor,
        input:  &Tensor,
        dim:    i64,
        output: &Tensor) -> Tensor {
    
    todo!();
        /*
            if (input.numel() <= 1) {
        return grad;
      }
      dim = maybe_wrap_dim(dim, input.dim());
      const i64 dim_size = input.size(dim);
      if (dim_size == 1) {
        return grad;
      }

      // To enable complex support.
      // From this line on `input_conj` and output_conj`
      // are interchangeable with `input` and `output`.
      auto input_conj = input.conj();
      auto output_conj = output.conj();

      const auto w = output_conj * grad;
      const auto is_zero = input == 0;
      if (!(is_zero.any().item<u8>())) {
        return reversed_cumsum(w, dim).div(input_conj);
      }

      // If we are not computing a second order gradient, we can use an
      // O(n) implementation. The derivative of this implementation is _not_
      // the second derivative of cumprod. As such, we fallback to a less efficient
      // O(n^2) implementation when GradMode::is_enabled().
      Tensor grad_input = zeros(input.sizes(), grad.options());
      if (!GradMode::is_enabled()) {
        // n.b. This could probably be implemented much faster with a kernel

        // From here on we need to use some mask gymnastics to
        // account for the tensorial dimensions
        // We do a cumsum of the zeros along the dimension.
        // For a vector is_zero = [False, True, False, True, False]
        // we would have cumsum = [0, 1, 1, 2, 2]
        // As such we have (in python code for simplicity)
        // The mask for the range [0, z1):
        // cumsum == 0
        // The indices of the first zero z1 and zeros when
        // there is no first zero:
        // indices = (cumsum == 1).max(dim, keepdim=True).indices
        // The mask for the first zero:
        // zeros_like(indices).scatter_(dim, indices, 1.) & cumsum == 1
        // Note that the logic_and with cumsum == 1 accounts
        // for the case when there is no first zero
        const auto cumsum = is_zero.cumsum(dim);

        // case k < z1
        // select everything before the first zero [0, z1)
        auto mask = cumsum == 0;
        // equiv to grad_input[mask] = deriv[grad]
        grad_input.masked_scatter_(mask,
            reversed_cumsum(w.masked_fill(~mask, 0.), dim).div_(input_conj).masked_select(mask));
        // select everything from the first zero to the second zero [z1, z2)
        mask = cumsum == 1;

        // case k = z1
        // We start by select the first zero [z1]
        // We locate the indices of the first zero using the max function
        // We then go from the indices to a mask index_fill_
        // When there is no zero in the slice, max will return the index 0.
        // To account for this, we need to do an intersection with mask,
        // which is true in the range [z1, z2)
        const auto first_zero_index = get<1>(mask.max(dim, /*keepdim*/ true));
        const auto first_zero_mask = zeros_like(mask)
                                      .scatter_(dim, first_zero_index, /*src*/ 1)
                                      .logical_and_(mask);

        // select everything between the first zero and the second zero (z1, z2)
        mask &= ~first_zero_mask;
        // here we compute
        // dy_j / dx_z1 = sum(cumprod(input[z1+1:z2] * grad[z1+1:z2])) * prod(output[z1-1])
        // relu_() necessary as gather does not support negative indices
        // finally, we do grad_input[z1] = dy_j / dx_z1
        grad_input.masked_scatter_(first_zero_mask,
                                   input_conj.masked_fill(~mask, 1.).cumprod(dim)
                                        .mul_(grad.masked_fill(cumsum != 1, 0.))
                                        .sum(dim, /*keepdim*/true)
                                        .mul_(gather(output_conj, dim, (first_zero_index - 1).relu_())
                                              .masked_fill_(first_zero_index == 0, 1.))
                                        .masked_select(first_zero_mask));
      } else { // GradMode::enabled()
        /*
        If the input is nonzero, we need to calculate the dy_j / dx_k
        by using the formula (2), called in the code omitted_products.

        The way the code calculates it is simply by noting that

        prod_{1 <= i <= j, i != k} x_i
            = (prod_{1 <= i <= k} x_i) * (prod_{k + 1 <= i <= j} x_i)

        the first term is calculated as prods_until_k, which since
        doesn't depend in j is easy to vectorize.

        The second term (indexed by j) is the cumulative product of
        x_{k+1}, x_{k+2}, ..., x_n, and it's named in the code
        prods_from_k_pkus_1, and it's calculated as a cumprod.

        In order to vectorize this properly, we need to add to
        omitted_products the dimensions where k > j, and therefore
        dy_j / dx_k = 0, which is done right after the assert.
        */

        auto ones_size = input.sizes().vec();
        ones_size[dim] = 1;
        const Tensor ones = ones({1}, grad.options()).expand(ones_size);
        Tensor prods_from_k_plus_1;
        Tensor omitted_products;
        for (const auto k : irange(dim_size)) {
          if (k == 0) {
            prods_from_k_plus_1 = cumprod(input_conj.slice(dim, k + 1), dim);
            omitted_products = cat({ones, prods_from_k_plus_1}, dim);
          } else if (k == dim_size - 1) {
            const Tensor prods_until_k = prod(input_conj.slice(dim, 0, k), dim, true);
            omitted_products = prods_until_k;
          } else {
            const Tensor prods_until_k = prod(input_conj.slice(dim, 0, k), dim, true);
            prods_from_k_plus_1 = cumprod(input_conj.slice(dim, k+1), dim);
            omitted_products = prods_until_k.expand_as(prods_from_k_plus_1) * prods_from_k_plus_1;
            omitted_products = cat({prods_until_k, omitted_products}, dim);
          }

          // At this point omitted_products is the same size
          // as input, except on the dimension dim where it's
          // dim_size - k
          TORCH_CHECK(omitted_products.size(dim) == dim_size - k);

          grad_input.select(dim, k).copy_(
              sum(grad.slice(dim, k) * omitted_products,dim));
        }
      }
      return grad_input;
        */
}

#[inline] pub fn isnan<T>(x: T) -> bool {

    todo!();
        /*
            return isnan(x);
        */
}

pub fn cummax_cummin_helper<T1, T2, Operation>(
    self_data:      *const T1,
    values_data:    *mut T1,
    indices_data:   *mut T2,
    self_dim_size:  i32,
    self_stride:    i32,
    values_stride:  i32,
    indices_stride: i32)  {

    todo!();
        /*
            Operation op;
          T1 out = self_data[0];
          int idx = 0;
          for(int i = 0; i < self_dim_size; i++) {
            T1 curr_elem = self_data[i*self_stride];
            if(isnan_(curr_elem) || (!isnan_(out) && op(curr_elem, out))) {
                out = self_data[i*self_stride];
                idx = i;
            }
            values_data[i*values_stride] = out;
            indices_data[i*indices_stride] = idx;
          }
        */
}


pub fn cummax_helper_cpu(
        self_:   &Tensor,
        values:  &mut Tensor,
        indices: &mut Tensor,
        dim:     i64)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool,
        self.scalar_type(), "cummax_cpu",
        [&] {
          native::tensor_dim_apply3<Scalar, i64>(self, values, indices, dim, cummax_cummin_helper<Scalar, i64, greater_equal<Scalar>>);
        });
        */
}


pub fn cummax_a(
    self_: &Tensor,
    dim:   Dimname) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return cummax(self, dimname_to_position(self, dim));
        */
}

pub fn cummax_b(
    self_: &Tensor,
    dim:   i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
            auto values = empty(self.sizes(), self.options());
      auto indices = empty(self.sizes(), self.options().dtype(kLong));
      cummax_out(values, indices, self, dim);
      return make_tuple(values, indices);
        */
}

pub fn cummin_helper_cpu(
    self_:   &Tensor,
    values:  &mut Tensor,
    indices: &mut Tensor,
    dim:     i64)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool,
        self.scalar_type(), "cummin_cpu",
        [&] {
          native::tensor_dim_apply3<Scalar, i64>(self, values, indices, dim, cummax_cummin_helper<Scalar, i64, less_equal<Scalar>>);
        });
        */
}

pub fn cummin_out_a<'a>(
    self_:   &Tensor,
    dim:     i64,
    values:  &mut Tensor,
    indices: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
            check_scalar_type_device_layout_equal(values, self);
      check_scalar_type_device_layout_equal(indices, empty({0}, self.options().dtype(kLong)));
      {
        NoNamesGuard guard;
        native::resize_output(values, self.sizes());
        native::resize_output(indices, self.sizes());
        if(self.dim() == 0) {
          values.fill_(self);
          indices.fill_(0);
        } else if(self.numel() != 0) {
          dim = maybe_wrap_dim(dim, self.dim());
          _cummin_helper(self, values, indices, dim);
        }
      }
      namedinference::propagate_names(values, self);
      namedinference::propagate_names(indices, self);
      return forward_as_tuple(values, indices);
        */
}

pub fn cummin_a(
    self_: &Tensor,
    dim:   i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
            auto values = empty(self.sizes(), self.options());
      auto indices = empty(self.sizes(), self.options().dtype(kLong));
      cummin_out(values, indices, self, dim);
      return make_tuple(values, indices);
        */
}

pub fn cummaxmin_backward(
    grad:    &Tensor,
    input:   &Tensor,
    indices: &Tensor,
    dim:     i64) -> Tensor {

    todo!();
        /*
            if (input.numel() == 0) {
        return input;
      }
      auto result = zeros(input.sizes(), input.options());
      return result.scatter_add_(dim, indices, grad);
        */
}

pub fn prepend_append_on_dim(
    self_:   &Tensor,
    prepend: &Option<Tensor>,
    append:  &Option<Tensor>,
    dim:     i64) -> Tensor {

    todo!();
        /*
            // Helper for diff that handles prepending and appending when at least one is present
      TORCH_INTERNAL_ASSERT(prepend.has_value() || append.has_value(), "either prepend or append must be have value");
      if (!prepend.has_value() && append.has_value()) {
        return cat({self, append.value()}, dim);
      } else if (prepend.has_value() && !append.has_value()) {
        return cat({prepend.value(), self}, dim);
      } else {
        return cat({prepend.value(), self, append.value()}, dim);
      }
        */
}

#[inline] pub fn diff_check_compatible_shape(
    self_: &Tensor,
    other: &Option<Tensor>,
    dim:   i64)  {
    
    todo!();
        /*
            // Helper for diff that checks whether the shape of the tensor to prepend or append
      // is compatible with that of input
      if (other.has_value()) {
        i64 wrapped_dim = maybe_wrap_dim(dim, self.dim(), false);

        TORCH_CHECK(
            other.value().dim() == self.dim(),
            "diff expects prepend or append to be the same dimension as input");

        for (int i = 0; i < other.value().dim(); i++) {
          TORCH_CHECK(
              other.value().size(i) == self.size(i) || i == wrapped_dim,
              "diff expects the shape of tensor to prepend or append to match that of"
              " input except along the differencing dimension;"
              " input.size(", i, ") = ", self.size(i), ", but got"
              " tensor.size(", i, ") = ", other.value().size(i));
        }
      }
        */
}

#[inline] pub fn diff_check(
    self_:   &Tensor,
    n:       i64,
    dim:     i64,
    prepend: &Option<Tensor>,
    append:  &Option<Tensor>)  {

    todo!();
        /*
            // Helper for diff that checks whether its parameters are valid
      TORCH_CHECK(
          n == 1,
          "diff only supports n = 1 currently. Please file an issue at"
          " https://github.com/pytorch/pytorch/issues/new?assignees=&labels=&template=feature-request.md"
          " if your use case requires supporting higher-order differences");

      TORCH_CHECK(
          self.dim() >= 1,
          "diff expects input to be at least one-dimensional");

      diff_check_compatible_shape(self, prepend, dim);
      diff_check_compatible_shape(self, append, dim);
        */
}

#[inline] pub fn diff_helper(
    self_: &Tensor,
    n:     i64,
    dim:   i64) -> Tensor {
    
    todo!();
        /*
            auto out_len = self.size(dim) - 1;
      if (self.dtype() == kBool) {
        return logical_xor(narrow(self, dim, 1, out_len), narrow(self, dim, 0, out_len));
      }
      return narrow(self, dim, 1, out_len) - narrow(self, dim, 0, out_len);
        */
}

pub fn diff(
    self_:   &Tensor,
    n:       i64,
    dim:     i64,
    prepend: &Option<Tensor>,
    append:  &Option<Tensor>) -> Tensor {

    todo!();
    /*
            diff_check(self, n, dim, prepend, append);
      if (!prepend.has_value() && !append.has_value()) {
        return diff_helper(self, n, dim);
      } else {
        auto a = prepend_append_on_dim(self, prepend, append, dim);
        return diff_helper(a, n, dim);
      }
        */
}


#[inline] pub fn diff_out_helper<'a>(
        self_:  &Tensor,
        n:      i64,
        dim:    i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            auto out_len = self.size(dim) - 1;
      if (self.dtype() == kBool) {
        return logical_xor_out(result, narrow(self, dim, 1, out_len), narrow(self, dim, 0, out_len));
      }
      return sub_out(result, narrow(self, dim, 1, out_len), narrow(self, dim, 0, out_len));
        */
}


pub fn diff_out<'a>(
        self_:   &Tensor,
        n:       i64,
        dim:     i64,
        prepend: &Option<Tensor>,
        append:  &Option<Tensor>,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            diff_check(self, n, dim, prepend, append);
      if (!prepend.has_value() && !append.has_value()) {
        return diff_out_helper(self, n, dim, result);
      } else {
        auto a = prepend_append_on_dim(self, prepend, append, dim);
        return diff_out_helper(a, n, dim, result);
      }
        */
}


pub fn pre_check_gradient(
        self_:        &Tensor,
        spacing_size: Option<i64>,
        dim:          Option<&[i32]>,
        edge_order:   i64)  {
    
    todo!();
        /*
            // Helper for gradient function to make sure input data satisfies prerequisites
      TORCH_CHECK(self.scalar_type() != ScalarType::Byte, "torch.gradient does not support uint8 input.");
      if (spacing_size.has_value() && !dim.has_value()) {
        TORCH_CHECK(spacing_size.value() == 1 || spacing_size.value() == self.dim(), "torch.gradient expected spacing to be unspecified, a scalar or a list of length ", self.dim(), "but got a list of length ", spacing_size.value());
      }
      if (spacing_size.has_value() && dim.has_value()) {
        TORCH_CHECK(spacing_size.value() == dim.value().size(), "torch.gradient expected spacing to be unspecified, a scalar or it's spacing and dim arguments to have the same length, but got a spacing argument of length ", spacing_size.value(), " and a dim argument of length ", dim.value().size(), "." );
      }
      // See discussion : https://github.com/pytorch/pytorch/issues/56036
      TORCH_CHECK(edge_order == 1, "torch.gradient only supports edge_order=1 currently. To request support for more edge_orders please file an issue here : https://github.com/pytorch/pytorch/issues/new?assignees=&labels=&template=feature-request.md");
      if (dim.has_value()) {
        // The following function get called to check whether dim argument satisfies prerequisites.
        // The output of the function is not used for the computation of gradient.
        dim_list_to_bitset(dim.value(), self.dim());
      }
        */
}


pub fn gradient_helper(
        self_:       &Tensor,
        coordinates: &[Tensor],
        dim:         &[i32],
        edge_order:  i64) -> Vec<Tensor> {
    
    todo!();
        /*
            for (const auto i : irange(coordinates.size())) {
        TORCH_CHECK(self.device() == coordinates[i].device(), "torch.gradient expected each tensor to be on the same device, but got devices ", self.device(), " and ", coordinates[i].device(), "!");
      }

      vector<Tensor> result;
      for (const auto i : irange(dim.size())) {
        TORCH_CHECK( coordinates[i].dim() == 1, "torch.gradient expected each element of spacing to have one dimension, but got an element with ", coordinates[i].dim(), " dimensions!");
        i64 direction = maybe_wrap_dim(dim[i], self.dim());
        vector<i64> shape(self.dim(),1);
        shape[ direction ] = -1;

        auto ax_dx = coordinates[i].diff(1,0);
        auto dx1 = slice(ax_dx, 0, 0, -1);
        auto dx2 = slice(ax_dx, 0, 1);
        auto a = (   -dx2    / (dx1*(dx1+dx2)) ).reshape(shape);
        auto b = ( (dx2-dx1) / (dx1*dx2)       ).reshape(shape);
        auto c = (    dx1    / (dx2*(dx1+dx2)) ).reshape(shape);

        auto center  = a*slice(self, direction, 0, -2) + b*slice(self, direction , 1, -1) + c*slice(self, direction ,2);
        auto prepend = (slice(self, direction, 1, 2  ) - slice(self, direction, 0, 1   )) / ax_dx[0]  ;
        auto append  = (slice(self, direction, -1    ) - slice(self, direction, -2, -1 )) / ax_dx[-1] ;
        result.emplace_back(prepend_append_on_dim(center, prepend, append, direction));
      }
      return result;
        */
}


pub fn gradient_helper_float(
        self_:      &Tensor,
        spacing:    &[Scalar],
        dim:        &[i32],
        edge_order: i64) -> Vec<Tensor> {
    
    todo!();
        /*
            vector<Tensor> result;
      for (const auto i : irange(dim.size())) {
          i64 direction = maybe_wrap_dim(dim[i], self.dim());
          auto ax_dx = spacing[i];
          auto center  = (slice(self,direction, 2   ) - slice(self, direction, 0, -2 ) ) / ax_dx;
          auto prepend = (slice(self,direction, 1, 2) - slice(self, direction, 0, 1  ) ) / ax_dx  ;
          auto append  = (slice(self,direction, -1  ) - slice(self, direction, -2, -1) ) / ax_dx ;
          result.emplace_back(prepend_append_on_dim(center/2, prepend, append, direction));
      }
      return result;
        */
}


pub fn gradient_dim_preprocess(
        self_: &Tensor,
        dim:   Option<i64>) -> Vec<i64> {
    
    todo!();
        /*
            // if gradient dim is provided as an integer, then we need to compute gradient only on this direction.
      // Moreover, if it's not provided at all, then we are interested in gradient for all directions.
      // Finally, if dim is provided as vector of ints, then it is not expected to be called by this function.
      if (dim.has_value()) {
        return vector<i64>{dim.value()};
      }

      vector<i64> axis(self.dim());
      iota(axis.begin(), axis.end(), 0);
      return axis;
        */
}


pub fn gradient_a(
        self_:       &Tensor,
        coordinates: &[Tensor],
        dim:         &[i32],
        edge_order:  i64) -> Vec<Tensor> {
    
    todo!();
        /*
            pre_check_gradient(self,
                           optional<i64>(coordinates.size()),
                           optional<IntArrayRef>(dim),
                           edge_order);
        return gradient_helper(self, coordinates, dim, edge_order);
        */
}


pub fn gradient_b(
        self_:       &Tensor,
        coordinates: &[Tensor],
        dim:         Option<i64>,
        edge_order:  i64) -> Vec<Tensor> {
    
    todo!();
        /*
            const auto processed_dim = gradient_dim_preprocess(self, dim);
      pre_check_gradient(self,
                         optional<i64>(coordinates.size()),
                         dim.has_value() ? optional<IntArrayRef>(processed_dim) : nullopt,
                         edge_order);
      return gradient_helper(self, coordinates, processed_dim, edge_order);
        */
}


pub fn gradient_c(
        self_:      &Tensor,
        spacing:    &[Scalar],
        dim:        &[i32],
        edge_order: i64) -> Vec<Tensor> {
    
    todo!();
        /*
            pre_check_gradient(self,
                         optional<i64>(spacing.size()),
                         optional<IntArrayRef>(dim),
                         edge_order);
      return gradient_helper_float(self, spacing, dim, edge_order);
        */
}


pub fn gradient_d(
        self_:      &Tensor,
        spacing:    &[Scalar],
        dim:        Option<i64>,
        edge_order: i64) -> Vec<Tensor> {
    
    todo!();
        /*
            const auto processed_dim = gradient_dim_preprocess(self, dim);
      pre_check_gradient(self,
                         optional<i64>(spacing.size()),
                         dim.has_value() ? optional<IntArrayRef>(processed_dim) : nullopt,
                         edge_order);
      return gradient_helper_float(self, spacing, processed_dim, edge_order);
        */
}


pub fn gradient_e(
        self_:      &Tensor,
        unit_size:  &Scalar,
        dim:        &[i32],
        edge_order: i64) -> Vec<Tensor> {
    
    todo!();
        /*
            // When spacing is given as scalar, while dim is given as IntArrayRef, scalar value need to
      // be taken as unit size at every given dimension element of - dim.
      vector<Scalar> spacing(dim.size(), unit_size);
      pre_check_gradient(self,
                         optional<i64>(spacing.size()),
                         optional<IntArrayRef>(dim),
                         edge_order);
      return gradient_helper_float(self, spacing, dim, edge_order);
        */
}


pub fn gradient_f(
        self_:      &Tensor,
        unit_size:  &Option<Scalar>,
        dim:        Option<i64>,
        edge_order: i64) -> Vec<Tensor> {
    
    todo!();
        /*
            const auto processed_dim = gradient_dim_preprocess(self, dim);
      // When unit_size not provided, it is always assumed to be equal to 1.
      // When dim has integer value it implies we are looking for gradient in the specific direction, however when
      // it is not provided, it means we are interested to find gradient in all directions.
      vector<Scalar> spacing(dim.has_value() ? 1 : self.dim(),
                                  unit_size.has_value() ? unit_size.value() : 1.0) ;
      pre_check_gradient(self,
                         unit_size.has_value() ?  optional<i64>(spacing.size()) : nullopt,
                         dim.has_value() ? optional<IntArrayRef>(processed_dim) : nullopt,
                         edge_order);
      return gradient_helper_float(self, spacing, processed_dim, edge_order);
        */
}


pub fn gradient_g(
        self_:      &Tensor,
        dim:        &[i32],
        edge_order: i64) -> Vec<Tensor> {
    
    todo!();
        /*
            vector<Scalar> spacing(dim.size(), 1.0) ;
      pre_check_gradient(self,
                         optional<i64>(spacing.size()),
                         optional<IntArrayRef>(dim),
                         edge_order);
      return gradient_helper_float(self, spacing, dim, edge_order);
        */
}



// ALL REDUCE #################################################################

#[inline] pub fn get_dtype_from_result(
        result: &mut Tensor,
        dtype:  Option<ScalarType>) -> ScalarType {
    
    todo!();
        /*
            TORCH_CHECK(result.defined(), "Cannot create a new tensor inside a reduction op. You likely tried to call an operator with an out argument but the out argument was an undefined tensor.");
      if (dtype.has_value()) {
        return dtype.value();
      } else {
        return result.scalar_type();
      }
        */
}


#[inline] pub fn get_dtype_from_self(
        self_:            &Tensor,
        dtype:            Option<ScalarType>,
        promote_integers: bool) -> ScalarType {
    
    todo!();
        /*
            if (dtype.has_value()) {
        return dtype.value();
      }
      ScalarType src_type = self.scalar_type();
      if (promote_integers && isIntegralType(src_type, /*includeBool=*/true)) {
        return kLong;
      }
      return src_type;
        */
}


pub fn sum_out_a<'a>(
        self_:     &Tensor,
        dim:       &[i32],
        keepdim:   bool,
        opt_dtype: Option<ScalarType>,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            ScalarType dtype = get_dtype_from_result(result, opt_dtype);
      auto iter = make_reduction("sum", result, self, dim, keepdim, dtype);
      if (iter.numel() == 0) {
        result.zero_();
      } else {
        sum_stub(iter.device_type(), iter);
      }
      return result;
        */
}


pub fn sum_a(
        self_: &Tensor,
        dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            return native::sum(self, vector<i64>{}, false, dtype);
        */
}


pub fn sum_b(
        self_:     &Tensor,
        dim:       &[i32],
        keepdim:   bool,
        opt_dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
      Tensor result = create_reduction_result(self, dim, keepdim, dtype);
      return native::sum_out(self, dim, keepdim, dtype, result);
        */
}


pub fn sum_c(
        self_:   &Tensor,
        dim:     &[Dimname],
        keepdim: bool,
        dtype:   Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            return sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
        */
}

pub fn sum_out_b<'a>(
    self_:     &Tensor,
    dim:       &[Dimname],
    keepdim:   bool,
    opt_dtype: Option<ScalarType>,
    result:    &mut Tensor) -> &'a mut Tensor {

    todo!();
        /*
            return sum_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
        */
}

pub fn nansum_out<'a>(
        self_:     &Tensor,
        dim:       &[i32],
        keepdim:   bool,
        opt_dtype: Option<ScalarType>,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!isComplexType(self.scalar_type()), "nansum does not support complex inputs");
      // For integral types, use existing sum as
      // integral types don't have `Nan`.
      if (isIntegralType(self.scalar_type(), true)){
        return sum_out(result, self, dim, keepdim, opt_dtype);
      }

      ScalarType dtype = get_dtype_from_result(result, opt_dtype);
      auto iter = make_reduction("nansum", result, self, dim, keepdim, dtype);
      if (iter.numel() == 0) {
        result = result.zero_();
      } else {
        nansum_stub(iter.device_type(), iter);
      }
      return result;
        */
}


pub fn nansum_a(
        self_: &Tensor,
        dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            return native::nansum(self, vector<i64>{}, false, dtype);
        */
}

pub fn nansum_b(
        self_:     &Tensor,
        dim:       &[i32],
        keepdim:   bool,
        opt_dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
      Tensor result = create_reduction_result(self, dim, keepdim, dtype);
      return native::nansum_out(self, dim, keepdim, dtype, result);
        */
}


pub fn prod_out_impl<'a>(
        result:    &mut Tensor,
        self_:     &Tensor,
        dim:       &[i32],
        keepdim:   bool,
        opt_dtype: Option<ScalarType>) -> &'a mut Tensor {
    
    todo!();
        /*
            ScalarType dtype = get_dtype_from_result(result, opt_dtype);
      auto iter = make_reduction("prod", result, self, dim, keepdim, dtype);
      if (iter.numel() == 0) {
        result.fill_(1);
      } else {
        prod_stub(iter.device_type(), iter);
      }
      return result;
        */
}

/**
  | -----------
  | @note
  | 
  | this could be implemented via diag and
  | sum, but this has perf problems, see
  | https://github.com/pytorch/pytorch/pull/47305,
  |
  */
pub fn trace_cpu(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      // Returns the ScalarType of the self tensor if the tensor is non integral type
      // In the case, self is an integer type tensor, kLong is return since promote_integers
      // is set to true
      ScalarType dtype = get_dtype_from_self(self, nullopt, true);
      result = empty({}, self.options().dtype(dtype));
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX(self.scalar_type(), "trace", [&] {
        using accscalar_t = acc_type<Scalar, false>;
        accscalar_t sum = 0;
        const auto* t_data = self.data_ptr<Scalar>();

        i64 t_stride_0, t_stride_1, t_diag_size;

        TORCH_CHECK(self.dim() == 2, "trace: expected a matrix, but got tensor with dim ", self.dim());

        t_stride_0 = self.stride(0);
        t_stride_1 = self.stride(1);

        t_diag_size = min(self.size(0), self.size(1));
        for (i64 i = 0; i < t_diag_size; i++) {
          sum += t_data[i * (t_stride_0 + t_stride_1)];
        }

        if_constexpr<is_integral<accscalar_t>::value>(
          // all integer types get promoted to kLong
          [&] (auto _) { *result.data_ptr<i64>() = _(sum); },  // then-case, invalid for non-integral types
          [&] (auto _) { *result.data_ptr<Scalar>() = _(sum); }  // else-case, invalid for integral types
        );
      });

      return result;
        */
}

pub fn prod_a(
        self_:     &Tensor,
        dim:       i64,
        keepdim:   bool,
        opt_dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
      Tensor result = create_reduction_result(self, dim, keepdim, dtype);
      native::prod_out_impl(result, self, dim, keepdim, dtype);
      return result;
        */
}


pub fn prod_b(
        self_:     &Tensor,
        opt_dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
      Tensor result = create_reduction_result(self, {}, false, dtype);
      return native::prod_out_impl(result, self, {}, false, dtype);
        */
}

pub fn prod_c(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool,
        dtype:   Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            return prod(self, dimname_to_position(self, dim), keepdim, dtype);
        */
}

pub fn prod_out_a<'a>(
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool,
        dtype:   Option<ScalarType>,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::prod_out_impl(result, self, dim, keepdim, dtype);
        */
}

pub fn prod_out_b<'a>(
        self_:     &Tensor,
        dim:       Dimname,
        keepdim:   bool,
        opt_dtype: Option<ScalarType>,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return prod_out(result, self, dimname_to_position(self, dim), keepdim, opt_dtype);
        */
}

pub fn mean_out_cpu_gpu<'a>(
        self_:     &Tensor,
        dim:       &[i32],
        keepdim:   bool,
        opt_dtype: Option<ScalarType>,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            ScalarType scalarType = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
      TORCH_CHECK(
          isFloatingType(scalarType) || isComplexType(scalarType),
          "Can only calculate the mean of floating types. Got ",
          toString(scalarType),
          " instead.");
      ScalarType dtype = get_dtype_from_result(result, opt_dtype);
      // TODO: the TensorIterator reduction implementation of mean
      // (mean_kernel_impl()) is unvectorized and leads to very poor performance
      // for production workloads. Once that's fixed, the following code can be used
      // in lieu of the sum + divide implementation below.
      if (self.device().is_cpu()) {
        i64 dim_prod = 1;
        if (dim.size() == 0 || self.ndimension() == 0) {
          dim_prod = self.numel();
        } else {
          for (auto d : dim) {
            dim_prod *= self.size(d);
          }
        }
        sum_out(result, self, dim, keepdim, dtype).div_(dim_prod);
        return result;
      }

      auto iter = make_reduction("mean", result, self, dim, keepdim, dtype);
      if (iter.numel() == 0) {
        result.fill_(numeric_limits<double>::quiet_NaN());
      } else {
        mean_stub(iter.device_type(), iter);
      }
      return result;
        */
}

pub fn mean_cpu_gpu_a(
        self_: &Tensor,
        dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            return native::mean_cpu_gpu(self, IntArrayRef{}, false, dtype);
        */
}

pub fn mean_cpu_gpu_b(
        self_:     &Tensor,
        dim:       &[i32],
        keepdim:   bool,
        opt_dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
      Tensor result = create_reduction_result(self, dim, keepdim, dtype);
      return native::mean_out_cpu_gpu(self, dim, keepdim, dtype, result);
        */
}


pub fn mean(
        self_:   &Tensor,
        dim:     &[Dimname],
        keepdim: bool,
        dtype:   Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            return mean(self, dimnames_to_positions(self, dim), keepdim, dtype);
        */
}


pub fn mean_out<'a>(
        self_:     &Tensor,
        dim:       &[Dimname],
        keepdim:   bool,
        opt_dtype: Option<ScalarType>,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return mean_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
        */
}


pub fn squeeze_multiple(
        self_: &Tensor,
        dims:  &[i32]) -> Tensor {
    
    todo!();
        /*
            int ndims = self.sizes().size();
      auto dims_to_squeeze = dim_list_to_bitset(dims, ndims);
      Tensor result = self;
      for (int i = ndims - 1; i >= 0; --i) {
        if (dims_to_squeeze[i]) {
          result = result.squeeze(i);
        }
      }
      return result;
        */
}


pub fn logsumexp_out_impl<'a>(
        result:  &mut Tensor,
        self_:   &Tensor,
        dims:    &[i32],
        keepdim: bool) -> &'a mut Tensor {
    
    todo!();
        /*
            // can't take max of empty tensor
      if (self.numel() != 0) {
        auto maxes = amax(self, dims, true);
        auto maxes_squeezed = (keepdim ? maxes : squeeze_multiple(maxes, dims));
        maxes_squeezed.masked_fill_(maxes_squeezed.abs() == INFINITY, 0);
        sum_out(result, (self - maxes).exp_(), dims, keepdim);
        result.log_().add_(maxes_squeezed);
      } else {
        sum_out(result, exp(self), dims, keepdim);
        result.log_();
      }
      return result;
        */
}


pub fn logsumexp_out<'a>(
        self_:   &Tensor,
        dims:    &[i32],
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            {
        NoNamesGuard guard;
        logsumexp_out_impl(result, self, dims, keepdim);
      }
      namedinference::propagate_names_for_reduction(result, self, dims, keepdim);
      return result;
        */
}


pub fn logsumexp_a(
        self_:   &Tensor,
        dims:    &[i32],
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return native::logsumexp_out(self, dims, keepdim, result);
        */
}


pub fn logsumexp_b(
        self_:   &Tensor,
        dims:    &[Dimname],
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            return logsumexp(self, dimnames_to_positions(self, dims), keepdim);
        */
}

pub fn logsumexp_ouw_With_dims<'a>(
    self_:   &Tensor,
    dims:    &[Dimname],
    keepdim: bool,
    result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return logsumexp_out(result, self, dimnames_to_positions(self, dims), keepdim);
        */
}

pub fn norm_out_a<'a>(
        result:    &mut Tensor,
        self_:     &Tensor,
        opt_p:     &Option<Scalar>,
        dim:       &[i32],
        keepdim:   bool,
        opt_dtype: Option<ScalarType>) -> &'a mut Tensor {
    
    todo!();
        /*
            auto p = opt_p.value_or(2.0).to<double>();
      TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
                  "norm only supports CPU and CUDA device types, but got: ", self.device().type());
      TORCH_CHECK(self.layout() == Layout::Strided,
                  "norm only supports strided layout, but got: ", self.layout());

      ScalarType in_dtype = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
      TORCH_CHECK(
          isFloatingType(in_dtype) || isComplexType(in_dtype),
          "Can only calculate the norm of floating point and complex dtypes. Got ",
          toString(in_dtype),
          " instead.");

      ScalarType out_dtype = result.defined() ? result.scalar_type() : (opt_dtype.has_value() ? opt_dtype.value() : toValueType(self.scalar_type()));

    // omit in_dtype in the following call, to avoid make_reduction explicitly casting input to out_dtype
      auto iter = isComplexType(self.scalar_type()) ?
          make_reduction("norm", result, self, dim, keepdim, in_dtype, out_dtype) :
          make_reduction("norm", result, self, dim, keepdim, out_dtype);

      if (iter.numel() == 0) {
        result.zero_();
      } else {
        norm_stub(iter.device_type(), iter, p);
      }
      return result;
        */
}


#[inline] pub fn norm_a(
        self_: &Tensor,
        p:     &Scalar) -> Tensor {
    
    todo!();
        /*
            if (self.is_sparse()) {
        // Sparse tensors need a different implementation because their values
        // are accessed with a different API than strided tensors
        return native_norm(self, p);
      } else {
        TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
                    "norm only supports CPU AND CUDA device type, got: ", self.device().type());
        TORCH_CHECK(self.layout() == Layout::Strided,
                    "norm only supports strided layout, got: ", self.layout());
        TORCH_CHECK(isFloatingType(self.scalar_type()) || isComplexType(self.scalar_type()),
                    "norm only supports floating-point dtypes");

        ScalarType dtype = toValueType(self.scalar_type());
        Tensor result = create_reduction_result(self, IntArrayRef{}, false, dtype);
        return native::norm_out(result, self, p, IntArrayRef{}, false, nullopt);
      }
        */
}


pub fn norm_out_b<'a>(
        self_:   &Tensor,
        p:       &Option<Scalar>,
        dim:     &[i32],
        keepdim: bool,
        dtype:   ScalarType,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::norm_out(result, self, p, dim, keepdim, optional<ScalarType>(dtype));
        */
}


pub fn norm_out_c<'a>(
        self_:   &Tensor,
        p:       &Option<Scalar>,
        dim:     &[i32],
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::norm_out(result, self, p, dim, keepdim, nullopt);
        */
}


pub fn norm_b(
        self_:     &Tensor,
        p:         &Option<Scalar>,
        dim:       &[i32],
        keepdim:   bool,
        opt_dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            if (self.is_sparse()) {
        // Sparse tensors need a different implementation because their values
        // are accessed with a different API than strided tensors
        return native_norm(self, p, dim, keepdim, opt_dtype);
      } else {
        ScalarType out_dtype = value_or_else(opt_dtype, [&] {return toValueType(self.scalar_type());});
        Tensor result = create_reduction_result(self, dim, keepdim, out_dtype);
        return native::norm_out(result, self, p, dim, keepdim, opt_dtype);
      }
        */
}


pub fn norm_c(
        self_:   &Tensor,
        p:       &Option<Scalar>,
        dim:     &[i32],
        keepdim: bool,
        dtype:   ScalarType) -> Tensor {
    
    todo!();
        /*
            return native::norm(self, p, dim, keepdim, optional<ScalarType>(dtype));
        */
}


pub fn norm_d(
        self_: &Tensor,
        p:     &Option<Scalar>,
        dtype: ScalarType) -> Tensor {
    
    todo!();
        /*
            return native::norm(self, p, IntArrayRef{}, false, optional<ScalarType>(dtype));
        */
}


pub fn norm_e(
        self_:   &Tensor,
        p:       &Option<Scalar>,
        dim:     &[i32],
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            return native::norm(self, p, dim, keepdim, nullopt);
        */
}

/**
  | leave it so we support sparse tensors
  |
  */
pub fn norm_f(
        self_: &Tensor,
        p:     &Scalar) -> Tensor {
    
    todo!();
        /*
            return native::_norm(self, p);
        */
}

/**
  | Note [all, any : uint8 compatibility]:
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | For NumPy comptability, `all` and `any` return
  |
  | Tensor of dtype `bool`. However for
  | compatibility reason, for `uint8`, they return
  | Tensor of same dtype `uint8`.
  |
  | Reference:
  | https://github.com/pytorch/pytorch/pull/47878#issuecomment-747108561
  |
  */
#[inline] pub fn all_a<'a>(
        result: &mut Tensor,
        iter:   &mut TensorIterator) -> &'a mut Tensor {
    
    todo!();
        /*
            if (iter.numel() == 0) {
        result.fill_(1);
      } else {
        and_stub(iter.device_type(), iter);
      }

      return result;
        */
}

pub fn all_b(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
                  "all only supports CPU AND CUDA device type, got: ", self.device().type());
      TORCH_CHECK(self.layout() == Layout::Strided,
                  "all only supports strided layout, got: ", self.layout());

      // Refer [all, any : uint8 compatibility]
      Tensor result;
      ScalarType out_dtype;
      if (self.scalar_type() == ScalarType::Byte){
        result = empty({0}, self.options());
        out_dtype = self.scalar_type();
      } else {
        result = empty({0}, self.options().dtype(kBool));
        out_dtype = ScalarType::Bool;
      }

      if (self.is_cuda()) {
        // As CUDA supports dynamic type casting, we use this overload of
        // `make_reduction`, which doesn't cast input to the result type i.e. kBool.,
        // otherwise we use the overload below which casts the input to kBool (which is
        // an extra operation).
        auto iter = make_reduction(
            "all", result, self, {}, false, self.scalar_type(), out_dtype);
        return _all(result, iter);
      }
      auto iter =
          make_reduction("all", result, self, {}, false, /*out_dtype=*/out_dtype);
      return _all(result, iter);
        */
}


pub fn all_c(
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            // Refer [all, any : uint8 compatibility]
      Tensor result;
      if (self.scalar_type() == ScalarType::Byte){
        result = empty({0}, self.options());
      } else {
        result = empty({0}, self.options().dtype(kBool));
      }

      return native::all_out(self, dim, keepdim, result);
        */
}


pub fn all_out_a<'a>(
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
                  "all only supports CPU AND CUDA device type, got: ", self.device().type());
      TORCH_CHECK(self.layout() == Layout::Strided,
                  "all only supports strided layout, got: ", self.layout());
      // Refer [all, any : uint8 compatibility]
      TORCH_CHECK(result.scalar_type() == ScalarType::Bool || result.scalar_type() == ScalarType::Byte,
                  "all only supports bool tensor for result, got: ", result.scalar_type());

      auto out_dtype = result.scalar_type();
      dim = maybe_wrap_dim(dim, self.dim());
      if (_dimreduce_return_trivial(result, self, 1, dim, keepdim)) {
        return result;
      } else {
        if (self.is_cuda()) {
          // As CUDA supports dynamic type casting, we use this overload of
          // `make_reduction`, which doesn't cast input to the result type i.e. kBool.,
          // otherwise we use the overload below which casts the input to kBool (which is
          // an extra operation).
          auto iter = make_reduction(
              "all", result, self, dim, keepdim, self.scalar_type(), out_dtype);
          return _all(result, iter);
        }
        auto iter =
            make_reduction("all", result, self, dim, keepdim, /*out_dtype=*/out_dtype);
        return _all(result, iter);
      }
        */
}


#[inline] pub fn any_a<'a>(
        result: &mut Tensor,
        iter:   &mut TensorIterator) -> &'a mut Tensor {
    
    todo!();
        /*
            if (iter.numel() == 0) {
        result.fill_(0);
      } else {
        or_stub(iter.device_type(), iter);
      }

      return result;
        */
}


pub fn any_b(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
                  "any only supports CPU AND CUDA device type, got: ", self.device().type());
      TORCH_CHECK(self.layout() == Layout::Strided || self.layout() == Layout::Sparse,
                  "any only supports strided AND sparse layout, got: ", self.layout());

      // Refer [all, any : uint8 compatibility]
      Tensor result;
      ScalarType out_dtype;
      if (self.scalar_type() == ScalarType::Byte){
        result = empty({0}, self.options());
        out_dtype = self.scalar_type();
      } else {
        result = empty({0}, self.options().dtype(kBool));
        out_dtype = ScalarType::Bool;
      }

      if (self.is_cuda()) {
        // As CUDA supports dynamic type casting, we use this overload of
        // `make_reduction`, which doesn't cast input to the result type i.e. kBool.,
        // otherwise we use the overload below which casts the input to kBool (which is
        // an extra operation).
        auto iter = make_reduction(
            "any", result, self, {}, false, self.scalar_type(), out_dtype);
        return _any(result, iter);
      }
      auto iter =
          make_reduction("any", result, self, {}, false, /*out_dtype=*/out_dtype);
      return _any(result, iter);
        */
}


pub fn any_c(
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            // Refer [all, any : uint8 compatibility]
      Tensor result;
      if (self.scalar_type() == ScalarType::Byte){
        result = empty({0}, self.options());
      } else {
        result = empty({0}, self.options().dtype(kBool));
      }

      return native::any_out(self, dim, keepdim, result);
        */
}


pub fn any_out_a<'a>(
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
                  "any only supports CPU AND CUDA device type, got: ", self.device().type());
      TORCH_CHECK(self.layout() == Layout::Strided,
                  "any only supports strided layout, got: ", self.layout());
      // Refer [all, any : uint8 compatibility]
      TORCH_CHECK(result.scalar_type() == ScalarType::Bool || result.scalar_type() == ScalarType::Byte,
                  "any only supports bool tensor for result, got: ", result.scalar_type());

      auto out_dtype = result.scalar_type();
      dim = maybe_wrap_dim(dim, self.dim());
      if (_dimreduce_return_trivial(result, self, 0, dim, keepdim)) {
        return result;
      } else {
        if (self.is_cuda()) {
          // As CUDA supports dynamic type casting, we use this overload of
          // `make_reduction`, which doesn't cast input to the result type i.e. kBool.,
          // otherwise we use the overload below which casts the input to kBool (which is
          // an extra operation).
          auto iter = make_reduction(
              "any", result, self, dim, keepdim, self.scalar_type(), out_dtype);
          return _any(result, iter);
        }
        auto iter =
            make_reduction("any", result, self, dim, keepdim, /*out_dtype=*/out_dtype);
        return _any(result, iter);
      }
        */
}


pub fn amin_out<'a>(
        self_:   &Tensor,
        dim:     &[i32],
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.scalar_type() == result.scalar_type(), "Expected the dtype for input and out to match, but got ",
                  self.scalar_type(), " for input's dtype and ",  result.scalar_type(), " for out's dtype.");
      if (self.numel() == 0) {
        zero_numel_check_dims(self, dim, "amin()");
      }

      auto iter = make_reduction("amin", result, self, dim, keepdim, self.scalar_type());
      if (iter.numel() != 0) {
        min_values_stub(iter.device_type(), iter);
      }
      return result;
        */
}


pub fn amin(
        self_:   &Tensor,
        dim:     &[i32],
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return amin_out(result, self, dim, keepdim);
        */
}


pub fn amax_out<'a>(
        self_:   &Tensor,
        dim:     &[i32],
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.scalar_type() == result.scalar_type(), "Expected the dtype for input and out to match, but got ",
                  self.scalar_type(), " for input's dtype and ",  result.scalar_type(), " for out's dtype.");
      if (self.numel() == 0) {
        zero_numel_check_dims(self, dim, "amax()");
      }

      auto iter = make_reduction("amax", result, self, dim, keepdim, self.scalar_type());
      if (iter.numel() != 0) {
        max_values_stub(iter.device_type(), iter);
      }
      return result;
        */
}


pub fn amax(
        self_:   &Tensor,
        dim:     &[i32],
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return amax_out(result, self, dim, keepdim);
        */
}


pub fn argmax_out<'a>(
        self_:   &Tensor,
        dim:     Option<i64>,
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            MaybeOwned<Tensor> in;
      if (dim) {
        auto sizes = self.sizes();
        zero_numel_check_dims(self, dim.value(), "argmax()");

        auto wrap_dim = maybe_wrap_dim(dim.value(), self.dim());
        if (sizes[wrap_dim] == 1) {
          if (keepdim) {
            result = zeros(sizes, self.options().dtype(kLong));
          } else {
            auto sizes_vec = sizes.vec();
            sizes_vec.erase(sizes_vec.begin() + wrap_dim);
            result = zeros(sizes_vec, self.options().dtype(kLong));
          }
          return result;
        }
        in = MaybeOwned<Tensor>::borrowed(self);
      } else {
        TORCH_CHECK_INDEX(self.numel() != 0, "argmax_out(): Expected reduction dim to be specified for input.numel() == 0.");
        in = MaybeOwned<Tensor>::owned(self.reshape({-1}));
        keepdim = false;
      }
      auto itr = make_reduction("argmax", result, *in, dim.value_or(0), keepdim,
          self.scalar_type(), kLong);
      if (itr.numel() != 0) {
        argmax_stub(itr.device_type(), itr);
      }
      return result;
        */
}


pub fn argmax(
        self_:    &Tensor,
        dim:      Option<i64>,
        keepdims: bool) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options().dtype(kLong));
      return native::argmax_out(self, dim, keepdims, result);
        */
}


pub fn argmin_out<'a>(
        self_:   &Tensor,
        dim:     Option<i64>,
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            MaybeOwned<Tensor> in;
      if (dim) {
        auto sizes = self.sizes();
        zero_numel_check_dims(self, dim.value(), "argmin()");

        auto wrap_dim = maybe_wrap_dim(dim.value(), self.dim());
        if (sizes[wrap_dim] == 1) {
          if (keepdim) {
            result = zeros(sizes, self.options().dtype(kLong));
          } else {
            auto sizes_vec = sizes.vec();
            sizes_vec.erase(sizes_vec.begin() + wrap_dim);
            result = zeros(sizes_vec, self.options().dtype(kLong));
          }
          return result;
        }
        in = MaybeOwned<Tensor>::borrowed(self);
      } else {
        TORCH_CHECK_INDEX(self.numel() != 0, "argmin_out(): Expected reduction dim to be specified for input.numel() == 0.");
        in = MaybeOwned<Tensor>::owned(self.reshape({-1}));
        keepdim = false;
      }
      auto itr = make_reduction("argmin", result, *in, dim.value_or(0), keepdim,
          self.scalar_type(), kLong);
      if (itr.numel() != 0) {
        argmin_stub(itr.device_type(), itr);
      }
      return result;
        */
}


pub fn argmin(
        self_:    &Tensor,
        dim:      Option<i64>,
        keepdims: bool) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options().dtype(kLong));
      return native::argmin_out(self, dim, keepdims, result);
        */
}


pub fn std_var_all_cpu(
        self_:      &Tensor,
        correction: i64,
        take_sqrt:  bool) -> f64 {
    
    todo!();
        /*
            const auto dtype = self.scalar_type();
      TORCH_CHECK(dtype == kDouble || dtype == kFloat,
                  "std_var_all: Unsupported dtype ", dtype);

      auto mean = self.mean().item<double>();
      auto iter = TensorIteratorConfig()
          .add_input(self)
          .build();

      auto reduction = [&](i64 begin, i64 end, double thread_sum) {
        AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "std_var_all_cpu", [&] {
          iter.serial_for_each([&] (char** data, const i64* strides, i64 size0, i64 size1) {
            const double local_mean = mean;
            const i64 inner_stride = strides[0];
            const i64 outer_stride = strides[1];

            double local_sum = 0.0;
            for (i64 i = 0; i < size1; ++i) {
              const char* row_ptr = data[0] + outer_stride * i;
              for (i64 j = 0; j < size0; ++j) {
                const auto ptr = reinterpret_cast<const Scalar*>(row_ptr + inner_stride * j);
                auto dx = (static_cast<double>(*ptr) - local_mean);
                local_sum += dx * dx;
              }
            }
            thread_sum += local_sum;
          }, {begin, end});
        });

        return thread_sum;
      };

      // ((x - mean)**2).sum()
      const double sum_dx2 = parallel_reduce(
          0, iter.numel(), internal::GRAIN_SIZE, 0.0, reduction, plus<>{});

      const auto var = [&] () __ubsan_ignore_float_divide_by_zero__ {
        return sum_dx2 / max(i64{0}, self.numel() - correction);
      }();
      const auto result = take_sqrt ? sqrt(var) : var;

      if (dtype == kFloat) {
        // Convert to infinity if out of range for a float.
        // Doing it now prevents checked_convert failing later
        return static_cast<float>(result);
      }
      return result;
        */
}


pub fn std_var_out<'a>(
        fname:          *const u8,
        result:         &mut Tensor,
        self_:          &Tensor,
        dim:            Option<&[i32]>,
        correction_opt: Option<i64>,
        keepdim:        bool,
        take_sqrt:      bool) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.device().is_cpu() || self.device().is_cuda(),
                  "std and var only supports tensors on a CPU or CUDA device, but got: ",
                  self.device().type());
      TORCH_CHECK(self.layout() == Layout::Strided,
                  "std and var only supports strided layout, got: ", self.layout());
      TORCH_CHECK(isFloatingType(self.scalar_type()) || isComplexType(self.scalar_type()),
                  "std and var only support floating point and complex dtypes");

      if (isComplexType(self.scalar_type())) {
        // For complex, calculate variance of real and imaginary components
        // seperately then add to get overall variance.
        ScalarType dtype = toValueType(get_dtype_from_result(result, {}));
        Tensor real_in = real(self);
        Tensor real_out = empty({0}, self.options().dtype(dtype));
        std_var_out(
            fname,
            real_out,
            real_in,
            dim,
            correction_opt,
            keepdim,
            /*take_sqrt=*/false);

        Tensor imag_in = imag(self);
        Tensor imag_out = empty({0}, self.options().dtype(dtype));
        std_var_out(
            fname,
            imag_out,
            imag_in,
            dim,
            correction_opt,
            keepdim,
            /*take_sqrt=*/false);

        add_out(result, real_out, imag_out);
        if (take_sqrt) {
          sqrt_out(result, result);
        }
        return result;
      }

      // Computation for floating point
      const auto correction = correction_opt.value_or(1);
      ScalarType dtype = get_dtype_from_result(result, {});
      auto iter = make_reduction(fname, result, self, dim, keepdim, dtype);

      if (iter.numel() == 0) {
        // Trivial reduction
        result.fill_(numeric_limits<double>::quiet_NaN());
        return result;
      } else if (
          result.numel() == 1 && iter.device_type() == kCPU &&
          iter.common_dtype() != kBFloat16 && iter.common_dtype() != kHalf) {
        // NOTE: CPU performance significantly regressed when attempting to port to
        // ATen,
        //   so all-reduce has a custom implementation.
        //   See https://github.com/pytorch/pytorch/pull/43858.
        result.fill_(std_var_all_cpu(self, correction, take_sqrt));
      } else {
        std_var_stub(iter.device_type(), iter, correction, take_sqrt);
      }
      return result;
        */
}


pub fn std_var_mean_out<'a>(
        fname:          *const u8,
        result1:        &mut Tensor,
        result2:        &mut Tensor,
        self_:          &Tensor,
        dim:            Option<&[i32]>,
        correction_opt: Option<i64>,
        keepdim:        bool,
        take_sqrt:      bool) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
            AT_ASSERT(result1.defined() && result2.defined());
      TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
                  fname, " only supports tensors on a CPU or CUDA device, got: ",
                  self.device().type());
      TORCH_CHECK(self.layout() == Layout::Strided,
                  fname, " only supports strided layout, got: ", self.layout());
      TORCH_CHECK(isFloatingType(self.scalar_type()) || isComplexType(self.scalar_type()),
                  fname, " only support floating point and complex dtypes");
      TORCH_CHECK(result1.scalar_type() == toValueType(result2.scalar_type()),
                  fname, " expected result1 to be real and match the precision of result2. Got ",
                  result1.scalar_type(), " and ", result2.scalar_type(), ".");

      if (isComplexType(self.scalar_type())) {
        // For complex, calculate for real and imaginary components seperately then combine as:
        // variance = var_real + var_imag
        // mean = mean_real + j * mean_imag
        ScalarType dtype = toValueType(get_dtype_from_result(result1, {}));
        Tensor real_in = real(self);
        Tensor real_out_var = empty({0}, self.options().dtype(dtype));
        Tensor real_out_mean = empty({0}, self.options().dtype(dtype));
        std_var_mean_out(
            fname,
            real_out_var,
            real_out_mean,
            real_in,
            dim,
            correction_opt,
            keepdim,
            /*take_sqrt=*/false);

        Tensor imag_in = imag(self);
        Tensor imag_out_var = empty({0}, self.options().dtype(dtype));
        Tensor imag_out_mean = empty({0}, self.options().dtype(dtype));
        std_var_mean_out(
            fname,
            imag_out_var,
            imag_out_mean,
            imag_in,
            dim,
            correction_opt,
            keepdim,
            /*take_sqrt=*/false);

        add_out(result1, real_out_var, imag_out_var);
        if (take_sqrt) {
          sqrt_out(result1, result1);
        }
        complex_out(result2, real_out_mean, imag_out_mean);
        return tuple<Tensor&, Tensor&>(result1, result2);
      }

      // Computation for floating point
      const auto correction = correction_opt.value_or(1);
      ScalarType dtype = get_dtype_from_result(result1, {});
      auto iter =
          make_reduction(fname, result1, result2, self, dim, keepdim, dtype);

      if (iter.numel() == 0) {
        // Trivial reduction
        result1.fill_(numeric_limits<double>::quiet_NaN());
        result2.fill_(numeric_limits<double>::quiet_NaN());
      } else {
        std_var_stub(iter.device_type(), iter, correction, take_sqrt);
      }
      return tuple<Tensor&, Tensor&>(result1, result2);
        */
}


pub fn var_mean_a(
        self_:    &Tensor,
        dim:      &[i32],
        unbiased: bool,
        keepdim:  bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return var_mean(self, /*dim=*/optional<IntArrayRef>(dim),
                          /*correction=*/i64{unbiased ? 1 : 0}, keepdim);
        */
}


pub fn std_mean_a(
        self_:    &Tensor,
        dim:      &[i32],
        unbiased: bool,
        keepdim:  bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return std_mean(self, /*dim=*/optional<IntArrayRef>(dim),
                          /*correction=*/i64{unbiased ? 1 : 0}, keepdim);
        */
}


pub fn std_mean_b(
        self_:    &Tensor,
        unbiased: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return std_mean(
          self, /*dim=*/nullopt, /*correction=*/i64{unbiased ? 1 : 0});
        */
}


pub fn var_mean_b(
        self_:    &Tensor,
        unbiased: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return var_mean(
          self, /*dim=*/nullopt, /*correction=*/i64{unbiased ? 1 : 0});
        */
}


/// Used in cuda/Normalization.cu
pub fn var_mean_out<'a>(
        result1:    &mut Tensor,
        result2:    &mut Tensor,
        self_:      &Tensor,
        dim:        &[i32],
        correction: i64,
        keepdim:    bool) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
            return std_var_mean_out(
          "var_mean", result1, result2, self, dim, correction, keepdim, false);
        */
}


pub fn options_to_value_type(opts: TensorOptions) -> TensorOptions {
    
    todo!();
        /*
            auto scalar_type = typeMetaToScalarType(opts.dtype());
      return opts.dtype(toValueType(scalar_type));
        */
}


pub fn var_mean_c(
        self_:      &Tensor,
        dim:        Option<&[i32]>,
        correction: Option<i64>,
        keepdim:    bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            Tensor result1 = empty({0}, options_to_value_type(self.options()));
      Tensor result2 = empty({0}, self.options());
      return std_var_mean_out(
          "var_mean", result1, result2, self, dim, correction, keepdim, false);
        */
}


pub fn std_mean_c(
        self_:      &Tensor,
        dim:        Option<&[i32]>,
        correction: Option<i64>,
        keepdim:    bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            Tensor result1 = empty({0}, options_to_value_type(self.options()));
      Tensor result2 = empty({0}, self.options());
      return std_var_mean_out(
          "std_mean", result1, result2, self, dim, correction, keepdim, true);
        */
}


pub fn var_a(
        self_:    &Tensor,
        unbiased: bool) -> Tensor {
    
    todo!();
        /*
            return var(
          self, /*dim=*/nullopt, /*correction=*/i64{unbiased ? 1 : 0});
        */
}


pub fn var_b(
        self_:    &Tensor,
        dim:      &[i32],
        unbiased: bool,
        keepdim:  bool) -> Tensor {
    
    todo!();
        /*
            return var(self, /*dim=*/optional<IntArrayRef>(dim),
                     /*correction=*/i64{unbiased ? 1 : 0}, keepdim);
        */
}


pub fn var_out_a<'a>(
        self_:    &Tensor,
        dim:      &[i32],
        unbiased: bool,
        keepdim:  bool,
        result:   &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return var_out(result, self, /*dim=*/optional<IntArrayRef>(dim),
                         /*correction=*/i64{unbiased ? 1 : 0}, keepdim);
        */
}


pub fn std_a(
        self_:    &Tensor,
        unbiased: bool) -> Tensor {
    
    todo!();
        /*
            return std(
          self, /*dim=*/nullopt, /*correction=*/i64{unbiased ? 1 : 0});
        */
}


pub fn std_b(
        self_:    &Tensor,
        dim:      &[i32],
        unbiased: bool,
        keepdim:  bool) -> Tensor {
    
    todo!();
        /*
            return std(self, /*dim=*/optional<IntArrayRef>(dim),
                     /*correction=*/i64{unbiased ? 1 : 0}, keepdim);
        */
}


pub fn std_out_a<'a>(
        self_:    &Tensor,
        dim:      &[i32],
        unbiased: bool,
        keepdim:  bool,
        result:   &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return std_out(result, self, /*dim=*/optional<IntArrayRef>(dim),
                         /*correction=*/i64{unbiased ? 1 : 0}, keepdim);
        */
}


pub fn std_c(
        self_:      &Tensor,
        dim:        Option<&[i32]>,
        correction: Option<i64>,
        keepdim:    bool) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, options_to_value_type(self.options()));
      return std_var_out("std", result, self, dim, correction, keepdim, true);
        */
}


pub fn std_out_b<'a>(
        self_:      &Tensor,
        dim:        Option<&[i32]>,
        correction: Option<i64>,
        keepdim:    bool,
        result:     &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return std_var_out("std", result, self, dim, correction, keepdim, true);
        */
}


pub fn var_out_b<'a>(
        self_:      &Tensor,
        dim:        Option<&[i32]>,
        correction: Option<i64>,
        keepdim:    bool,
        result:     &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return std_var_out("var", result, self, dim, correction, keepdim, false);
        */
}


pub fn var_c(
        self_:      &Tensor,
        dim:        Option<&[i32]>,
        correction: Option<i64>,
        keepdim:    bool) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, options_to_value_type(self.options()));
      return std_var_out("var", result, self, dim, correction, keepdim, false);
        */
}


pub fn std_d(
        self_:    &Tensor,
        dim:      &[Dimname],
        unbiased: bool,
        keepdim:  bool) -> Tensor {
    
    todo!();
        /*
            return std(self, dimnames_to_positions(self, dim), unbiased, keepdim);
        */
}


pub fn std_out_c<'a>(
        self_:    &Tensor,
        dim:      &[Dimname],
        unbiased: bool,
        keepdim:  bool,
        result:   &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return std_out(result, self, dimnames_to_positions(self, dim), unbiased, keepdim);
        */
}


pub fn var_d(
        self_:    &Tensor,
        dim:      &[Dimname],
        unbiased: bool,
        keepdim:  bool) -> Tensor {
    
    todo!();
        /*
            return var(self, dimnames_to_positions(self, dim), unbiased, keepdim);
        */
}


pub fn var_out_c<'a>(
        dim:      &[Dimname],
        unbiased: bool,
        keepdim:  bool,
        result:   &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return var_out(
          result, self, dimnames_to_positions(self, dim), unbiased, keepdim);
        */
}


pub fn var_mean_d(
        self_:    &Tensor,
        dim:      &[Dimname],
        unbiased: bool,
        keepdim:  bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return var_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
        */
}


pub fn std_mean_d(
        self_:    &Tensor,
        dim:      &[Dimname],
        unbiased: bool,
        keepdim:  bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return std_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
        */
}


pub fn std_e(
        self_:      &Tensor,
        dim:        &[Dimname],
        correction: Option<i64>,
        keepdim:    bool) -> Tensor {
    
    todo!();
        /*
            return std(self, dimnames_to_positions(self, dim), correction, keepdim);
        */
}


pub fn std_out_d<'a>(
        self_:      &Tensor,
        dim:        &[Dimname],
        correction: Option<i64>,
        keepdim:    bool,
        result:     &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return std_out(result, self, dimnames_to_positions(self, dim), correction, keepdim);
        */
}

pub fn var_e(
        self_:      &Tensor,
        dim:        &[Dimname],
        correction: Option<i64>,
        keepdim:    bool) -> Tensor {
    
    todo!();
        /*
            return var(self, dimnames_to_positions(self, dim), correction, keepdim);
        */
}


pub fn var_out_d<'a>(
        self_:      &Tensor,
        dim:        &[Dimname],
        correction: Option<i64>,
        keepdim:    bool,
        result:     &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return var_out(
          result, self, dimnames_to_positions(self, dim), correction, keepdim);
        */
}


pub fn var_mean_e(
        self_:      &Tensor,
        dim:        &[Dimname],
        correction: Option<i64>,
        keepdim:    bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return var_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
        */
}


pub fn std_mean_e(
        self_:      &Tensor,
        dim:        &[Dimname],
        correction: Option<i64>,
        keepdim:    bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return std_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
        */
}


pub fn norm_out_d<'a>(
        self_:   &Tensor,
        p:       &Option<Scalar>,
        dim:     &[Dimname],
        keepdim: bool,
        dtype:   ScalarType,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim, dtype);
        */
}


pub fn norm_out_e<'a>(
        self_:   &Tensor,
        p:       &Option<Scalar>,
        dim:     &[Dimname],
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim);
        */
}


pub fn norm_g(
        self_:   &Tensor,
        p:       &Option<Scalar>,
        dim:     &[Dimname],
        keepdim: bool,
        dtype:   ScalarType) -> Tensor {
    
    todo!();
        /*
            return norm(self, p, dimnames_to_positions(self, dim), keepdim, dtype);
        */
}


pub fn norm_h(
        self_:   &Tensor,
        p:       &Option<Scalar>,
        dim:     &[Dimname],
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            return norm(self, p, dimnames_to_positions(self, dim), keepdim);
        */
}


pub fn any_d(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("any");
        */
}


pub fn any_out_b<'a>(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("any");
        */
}


pub fn all_d(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("all");
        */
}


pub fn all_out_b<'a>(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("all");
        */
}


pub fn logcumsumexp_with_dim(
        self_: &Tensor,
        dim:   Dimname) -> Tensor {
    
    todo!();
        /*
            return logcumsumexp(self, dimname_to_position(self, dim));
        */
}

pub fn logcumsumexp_out_a<'a>(
        self_:  &Tensor,
        dim:    i64,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            check_scalar_type_device_layout_equal(result, self);
      {
        NoNamesGuard guard;
        _logcumsumexp_out(result, self.toType(result.scalar_type()), dim);
      }
      namedinference::propagate_names(result, self);
      return result;
        */
}

pub fn logcumsumexp_out_b<'a>(
        self_:  &Tensor,
        dim:    Dimname,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return logcumsumexp_out(result, self, dimname_to_position(self, dim));
        */
}

pub fn cumsum_mut<'a>(
        self_: &mut Tensor,
        dim:   Dimname,
        dtype: Option<ScalarType>) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::cumsum_(self, dimname_to_position(self, dim), dtype);
        */
}

pub fn cumprod_out_a<'a>(
    self_:  &Tensor,
    dim:    i64,
    dtype:  Option<ScalarType>,
    result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
      TORCH_CHECK(
          !dtype.has_value() || (result.scalar_type() == dtype.value()),
          "provided dtype must match dtype of result in cumprod. Got ",
          toString(result.scalar_type()),
          " and ",
          toString(dtype.value()),
          ".");
      {
        NoNamesGuard guard;
        _cumprod_out(result, self.toType(result.scalar_type()), dim);
      }
      namedinference::propagate_names(result, self);
      return result;
        */
}

pub fn cumprod_out_b<'a>(
    self_:  &Tensor,
    dim:    Dimname,
    dtype:  Option<ScalarType>,
    result: &mut Tensor) -> &'a mut Tensor {

    todo!();
        /*
            return cumprod_out(result, self, dimname_to_position(self, dim), dtype);
        */
}

pub fn cummax_out_a<'a>(
    self_:   &Tensor,
    dim:     i64,
    values:  &mut Tensor,
    indices: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {

    todo!();
        /*
            check_scalar_type_device_layout_equal(values, self);
      check_scalar_type_device_layout_equal(indices, empty({0}, self.options().dtype(kLong)));
      {
        NoNamesGuard guard;
        native::resize_output(values, self.sizes());
        native::resize_output(indices, self.sizes());
        if(self.dim() == 0) {
          values.fill_(self);
          indices.fill_(0);
        } else if(self.numel() != 0) {
          dim = maybe_wrap_dim(dim, self.dim());
          _cummax_helper(self, values, indices, dim);
        }
      }
      namedinference::propagate_names(values, self);
      namedinference::propagate_names(indices, self);
      return forward_as_tuple(values, indices);
        */
}

pub fn cummax_out_b<'a>(
    self_:   &Tensor,
    dim:     Dimname,
    values:  &mut Tensor,
    indices: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {

    todo!();
        /*
            return cummax_out(values, indices, self, dimname_to_position(self, dim));
        */
}

pub fn cummin_b(
    self_: &Tensor,
    dim:   Dimname) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return cummin(self, dimname_to_position(self, dim));
        */
}

pub fn cummin_out_b<'a>(
    self_:   &Tensor,
    dim:     Dimname,
    values:  &mut Tensor,
    indices: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {

    todo!();
        /*
            return cummin_out(values, indices, self, dimname_to_position(self, dim));
        */
}

pub fn dist(
    self_: &Tensor,
    other: &Tensor,
    p:     &Scalar) -> Tensor {
    
    todo!();
        /*
            return norm(self - other, p);
        */
}

pub fn cpu_equal(
    self_: &Tensor,
    other: &Tensor) -> bool {
    
    todo!();
        /*
            if (!namedinference::are_names_equal(
            self.unsafeGetTensorImpl(), other.unsafeGetTensorImpl())) {
        return false;
      }
      NoNamesGuard guard;
      TORCH_CHECK(self.device() == other.device(), "Cannot compare two tensors on "
                  "different devices. Got: ", self.device(), " and ", other.device());
      TORCH_CHECK(self.dtype() == other.dtype(),
                  "Expected object of scalar type ", self.dtype(), " but got scalar type ",
                  other.dtype(), " for argument 'other'");
      if (!self.is_same_size(other)) {
        return false;
      }
      atomic<bool> result{true};
      auto iter = TensorIteratorConfig()
        .add_input(self)
        .add_input(other)
        .allow_cpu_scalars(true)
        .promote_inputs_to_common_dtype(true)
        .build();

      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "equal_cpu", [&] {
        iter.for_each([&](char** data, const i64 *strides, i64 dim_size) {
          if (!result) {
              return;
          }
          char* self_data = data[0];
          char* other_data = data[1];
          for (i64 i = 0; i < dim_size; ++i) {
            if (*((Scalar*)self_data) != *((Scalar*)other_data)) {
              result = false;
              return;
            }
            self_data += strides[0];
            other_data += strides[1];
          }
        });
      });
      return result.load();
        */
}

/**
  | max(dim), min(dim), topk(dim), mode(dim),
  | are examples of reduction functions
  | that select values. value_selecting_reduction_backward
  | is the backward function for those operators;
  | it propagates the grad to the specific
  | value locations referred to at `indices`.
  |
  */
pub fn value_selecting_reduction_backward(
    grad:    &Tensor,
    dim:     i64,
    indices: &Tensor,
    sizes:   &[i32],
    keepdim: bool) -> Tensor {
    
    todo!();
        /*
            if (!keepdim && sizes.size() > 0) {
        auto grad_ = grad.unsqueeze(dim);
        auto indices_ = indices.unsqueeze(dim);
        return zeros(sizes, grad_.options()).scatter_(dim, indices_, grad_);
      }
      return zeros(sizes, grad.options()).scatter_(dim, indices, grad);
        */
}
