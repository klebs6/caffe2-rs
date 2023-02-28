crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/DistributionTemplates.h]

// ==================================================== Random ========================================================

/**
  | The purpose of `update_from` and `update_to` is
  | to find the closest valid i64 number that
  | can be used as actual `from`.
  |
  | The current implementation of `random_` uses
  | u64 arithmetics and casts the result to
  | the target dtype(Scalar).
  |
  | This casting can result in generating numbers
  | that happen to be greater or equal to `to`
  | value. For instance:
  |
  |    auto actual = Torchempty({3, 3}, Torchhalf);
  |    actual.random_(0, 65504);
  |
  | If random's u64 arithmetics produces 65503
  | as a random value after casting to Torchhalf it
  | becomes 65504 and violates the requirement that
  | random value must be less than `to`.
  |
  | To resolve this issue `update_from` and
  | `update_to` moves `from` to the left and `to`
  | to the right to the next closest value that
  | won't go outside [from, to) after casting to
  | the target dtype. For `to` = 65504 it moves
  | left for (1 << (log2(to) - 11 + 1)) = 32 and
  | becomes 65472, which is previous available
  | number for Torchhalf dtype.
  |
  */
pub fn update_from<Scalar>(from: i64) -> i64 {

    todo!();
        /*
            static_assert(
        is_floating_point<Scalar>::value ||
        is_same<Scalar, Half>::value ||
        is_same<Scalar, BFloat16>::value, "Scalar must be floating-point type");
      const auto from_plus_1 = static_cast<i64>(static_cast<Scalar>(from + 1));
      if (from_plus_1 < from) {
        i64 from_ = abs(from + 1);
        int n = 0;
        while (from_ >>= 1) ++n;
        from = from_plus_1 + (1LL << (n - numeric_limits<Scalar>::digits + 1));
      }
      return from;
        */
}

pub fn update_to<Scalar>(to: i64) -> i64 {

    todo!();
        /*
            static_assert(
        is_floating_point<Scalar>::value ||
        is_same<Scalar, Half>::value ||
        is_same<Scalar, BFloat16>::value, "Scalar must be floating-point type");
      const auto to_minus_1 = static_cast<i64>(static_cast<Scalar>(to - 1));
      if (to_minus_1 >= to) {
        i64 to_ = abs(to - 1);
        int n = 0;
        while (to_ >>= 1) ++n;
        to = to_minus_1 - (1LL << (n - numeric_limits<Scalar>::digits + 1));
      }
      return to;
        */
}


pub fn random_impl<random_kernel, RNG>(
        self_:     &mut Tensor,
        generator: Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            auto iter = TensorIterator::borrowing_nullary_op(self);
      random_kernel<RNG>()(iter, generator);
      return self;
        */
}

#[macro_export] macro_rules! check_out_of_bounds {
    ($var:ident, $name:ident, $min:ident, $max:ident, $dtype:ident) => {
        /*
        
          TORCH_CHECK(var >= min && var <= max, name , " is out of bounds for ", dtype); 
        */
    }
}


#[macro_export] macro_rules! warn_out_of_bounds {
    ($var:ident, $name:ident, $digits:ident, $dtype:ident) => {
        /*
        
          if (var < -(1LL << digits) || var > (1LL << digits)) { 
            TORCH_WARN(name , " is out of bounds [-(2^", digits, "), 2^", digits, "]. ", 
              "Due to precision limitations ", dtype, " can support discrete uniform distribution only within this range. ", 
              "This warning will become an error in version 1.7 release, please fix the code in advance"); 
          }
        */
    }
}

pub fn check_from_to_in_range(
        from:   i64,
        to_inc: i64,
        dtype:  TypeMeta)  {
    
    todo!();
        /*
            const auto scalar_type = typeMetaToScalarType(dtype);
      if (isFloatingType(scalar_type)) {
        AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, scalar_type, "check_random_fp_bounds", [&] {
          const auto min = static_cast<double>(numeric_limits<Scalar>::lowest());
          const auto max = static_cast<double>(Scalar::max);
          CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
          CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max, dtype);

          constexpr auto digits = numeric_limits<Scalar>::digits;
          WARN_OUT_OF_BOUNDS(from, "from", digits, dtype);
          WARN_OUT_OF_BOUNDS(to_inc, "to - 1", digits, dtype);
        });
      } else if (isIntegralType(scalar_type, /*includeBool=*/true)) {
        AT_DISPATCH_INTEGRAL_TYPES_AND(ScalarType::Bool, scalar_type, "check_random_integral_bounds", [&]() {
          const auto min = static_cast<i64>(numeric_limits<Scalar>::lowest());
          const auto max = static_cast<i64>(Scalar::max);
          CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
          CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max, dtype);
        });
      } else {
        TORCH_CHECK(false, "check_random_bounds handles only integral, floating-point and boolean types");
      }
        */
}

pub fn random_from_to_impl<random_from_to_kernel, RNG>(
        self_:     &mut Tensor,
        from:      i64,
        to_opt:    Option<i64>,
        generator: Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            u64 range = 0;
      auto iter = TensorIterator::borrowing_nullary_op(self);
      if (to_opt.has_value()) {
        // [from, to)
        i64 to = *to_opt;
        TORCH_CHECK(from < to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);
        if (isFloatingType(iter.dtype())) {
          AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, self.scalar_type(), "random_update_from_to", [&] {
            from = update_from<Scalar>(from);
            to = update_to<Scalar>(to);
            TORCH_CHECK(from < to, "random_ expects 'from' casted to dtype to be less than 'to' casted to dtype, but got from=", from, " >= to=", to);
          });
        }
        check_from_to_in_range(from, to - 1, self.dtype());
        range = static_cast<u64>(to) - static_cast<u64>(from);
        random_from_to_kernel<RNG>()(iter, range, from, generator);
      } else if (from != numeric_limits<i64>::lowest()) {
        // [from, i64::max]
        i64 to_inc = 0;
        if (isFloatingType(iter.dtype())) {
          AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, self.scalar_type(), "random_from_to_range_calc", [&] {
            constexpr i64 scalar_t_max = static_cast<i64>(1) << numeric_limits<Scalar>::digits;
            to_inc = scalar_t_max > i64::max ? i64::max : static_cast<i64>(scalar_t_max);
            from = update_from<Scalar>(from);
            TORCH_CHECK(from < to_inc, "random_ expects 'from' casted to dtype to be less than or equal to 'to_inc' casted to dtype, but got from=", from, " > to_inc=", to_inc);
          });
        } else if (isIntegralType(iter.dtype(), /*includeBool=*/true)) {
          AT_DISPATCH_INTEGRAL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "random_from_to_range_calc", [&] {
            if (is_same<Scalar, bool>::value) {
              to_inc = static_cast<i64>(true);
            } else {
              to_inc = static_cast<i64>(Scalar::max);
            }
          });
        } else {
          TORCH_CHECK(false, "random_from_to_impl handles only integral, floating-point and boolean types");
        }
        check_from_to_in_range(from, to_inc, self.dtype());
        range = static_cast<u64>(to_inc) - static_cast<u64>(from) + 1;
        random_from_to_kernel<RNG>()(iter, range, from, generator);
      } else {
        // [numeric_limits<i64>::lowest(), i64::max]
        // range = 2^64
        random_from_to_kernel<RNG>()(iter, generator);
      }
      return self;
        */
}

// ==================================================== Normal ========================================================

/**
  | This function computes broadcasted size of mean
  | and std, resize the output to the broadcasted
  | size if it was empty
  |
  | [Note] The following features will be
  | deprecated in version 1.6 release and function
  | signature will be changed after
  |
  |   When mean and std are not broadcastable but
  |   have same number of elements:
  |
  |     This function will resize the output to the
  |     size of mean if it was empty.
  |
  |     This function will reshape the std to the
  |     shape of mean.
  |
  |     This function will return true in
  |     deprecated case, false in broadcastable
  |     case and throw in all other cases before
  |     deprecation.
  |
  |     This function will not return and throw if
  |     mean and std are not broadcastable after
  |     deprecation
  |
  */
pub fn resize_output_for_normal(
    output: &mut Tensor,
    mean:   &Tensor,
    std:    &Tensor) -> bool {
    
    todo!();
        /*
            bool expandable = are_expandable(mean.sizes(), std.sizes());
      bool empty_output = output.numel() == 0;

      if (expandable) {
        auto shape = infer_size(mean.sizes(), std.sizes());
        TORCH_CHECK(
            empty_output || output.sizes().equals(shape),
            "inconsistent tensor, output size (", output.sizes(), ") is not the same as broadcasted mean and std size (", shape, ")");
        if (empty_output) {
          native::resize_(output, shape);
        }
        return false;
      }
      else {
        TORCH_CHECK(
            mean.numel() == std.numel(),
            "inconsistent tensor, std and mean are not broadcastable and have different number of elements, "
            "expected mean ", mean.sizes(), " and std ", std.sizes(), " to have same number of elements)");
        TORCH_CHECK(
            empty_output || output.sizes().equals(mean.sizes()),
            "inconsistent tensor, std and mean are not broadcastable, output size (", output.sizes(), ") is not the same as mean size (", mean.sizes(), ")");
        TORCH_WARN_ONCE(
            "std and mean have the same number of elements, but are not broadcastable. This was previously a "
            "supported mode of operation, but is now deprecated and the support will be removed in version 1.6 release. "
            "Note that the current implementation reshapes std to the shape of mean, which may be incur data copies. "
            "Please ensure that std and mean are broadcastable to avoid these issues.");
        if (empty_output) {
          native::resize_(output, mean.sizes());
        }
        return true;
      }
        */
}

pub fn normal_impl_f64_f64<normal_kernel, RNG>(
        self_: &mut Tensor,
        mean:  f64,
        std:   f64,
        gen:   Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std);
      if (self.is_complex()) {
        auto float_tensor = view_as_real(self);
        // variance for normal distribution of the real and imaginary values
        // is half of the input variance
        normal_kernel<RNG>()(float_tensor, mean, std/(sqrt(2)), gen);
      } else {
        normal_kernel<RNG>()(self, mean, std, gen);
      }
      return self;
        */
}

pub fn normal_out_impl_tensor_f64<normal_kernel, RNG>(
        output: &mut Tensor,
        mean:   &Tensor,
        std:    f64,
        gen:    Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            normal_impl_<normal_kernel, RNG>(output, 0, std, gen);
      output.add_(mean);
      return output;
        */
}

pub fn normal_out_impl_f64_tensor<normal_kernel, RNG>(
        output: &mut Tensor,
        mean:   f64,
        std:    &Tensor,
        gen:    Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            TORCH_CHECK(!std.is_complex(), "normal expects standard deviation to be non-complex");
      TORCH_CHECK(
        std.min().ge(0).item<bool>(),
        "normal expects all elements of std >= 0.0");
      normal_impl_<normal_kernel, RNG>(output, 0, 1, gen);
      auto mean_tensor = full({}, mean, output.options());
      // CUDA NB: addcmul_out copies the tensor to be added into the output.
      // Please look at aten/src/THC/generic/THCTensorMathPointwise.cu
      // The previous function here was addcmul_out(output, mean_tensor, output, std, 1);
      // The third argument is not a constant reference and hence the samples in output are overwritten.
      // Consequently, the computation performed is mean_tensor + mean_tensor * std instead of mean_tensor + output * std
      output.mul_(std).add_(mean_tensor);
      return output;
        */
}

pub fn normal_out_impl_tensor_tensor<normal_kernel, RNG>(
        output: &mut Tensor,
        mean:   &Tensor,
        std:    &Tensor,
        gen:    Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            TORCH_CHECK(!std.is_complex(), "normal expects standard deviation to be non-complex");
      TORCH_CHECK(
        std.min().ge(0).item<bool>(),
        "normal expects all elements of std >= 0.0");
      bool is_deprecated_th_impl = resize_output_for_normal(output, mean, std);
      normal_impl_<normal_kernel, RNG>(output, 0, 1, gen);
      // CUDA NB: addcmul_out copies the tensor to be added into the output.
      // Please look at aten/src/THC/generic/THCTensorMathPointwise.cu
      // The previous function here was addcmul_out(output, mean, output, std, 1);
      // The third argument is not a constant reference and hence the samples in output are overwritten.
      // Consequently, the computation performed is mean + mean * std instead of mean + output * std
      if (is_deprecated_th_impl) {
        output.mul_(std.reshape(mean.sizes())).add_(mean);
      }
      else {
        output.mul_(std).add_(mean);
      }
      return output;
        */
}


pub fn normal_impl_a<normal_kernel, RNG>(
        mean: &Tensor,
        std:  f64,
        gen:  Option<Generator>) -> Tensor {

    todo!();
        /*
            Tensor ret = empty_like(mean, MemoryFormat::Contiguous);
      normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
      return ret;
        */
}

pub fn normal_impl_b<normal_kernel, RNG>(
        mean: f64,
        std:  &Tensor,
        gen:  Option<Generator>) -> Tensor {

    todo!();
        /*
            Tensor ret = empty_like(std, MemoryFormat::Contiguous);
      normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
      return ret;
        */
}

pub fn normal_impl_tensor<normal_kernel, RNG>(
        mean: &Tensor,
        std:  &Tensor,
        gen:  Option<Generator>) -> Tensor {

    todo!();
        /*
            Tensor ret = empty({0}, mean.options(), MemoryFormat::Contiguous);
      normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
      return ret;
        */
}

// ==================================================== Uniform =======================================================

pub fn uniform_impl<uniform_kernel, RNG>(
        self_:     &mut Tensor,
        from:      f64,
        to:        f64,
        generator: Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            if (self.is_complex()) {
        auto float_tensor = view_as_real(self);
        uniform_impl_<uniform_kernel, RNG>(float_tensor, from, to, generator);
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, self.scalar_type(), "check_uniform_bounds", [&] {
          const auto dtype = self.dtype();
          const auto min = static_cast<double>(numeric_limits<Scalar>::lowest());
          const auto max = static_cast<double>(Scalar::max);
          CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
          CHECK_OUT_OF_BOUNDS(to, "to", min, max, dtype);
          TORCH_CHECK(from <= to, "uniform_ expects to return a [from, to) range, but found from=", from, " > to=", to);
          TORCH_CHECK((to - from) <= Scalar::max,
                "uniform_ expects to-from <= numeric_limits<", toString(self.scalar_type()),
                ">::max(), but found to=", to, " and from=", from,
                " which result in to-from to exceed the limit");
          from = min(max(from, min), max);
          to = max(min(to, max), min);
        });
        auto iter = TensorIterator::borrowing_nullary_op(self);
        uniform_kernel<RNG>()(iter, from, to, generator);
      }
      return self;
        */
}

// ================================================== LogNormal =======================================================

pub fn log_normal_impl<log_normal_kernel, RNG>(
        self_: &mut Tensor,
        mean:  f64,
        std:   f64,
        gen:   Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            TORCH_CHECK(std > 0.0, "log_normal_ expects std > 0.0, but found std=", std);
      auto iter = TensorIterator::borrowing_nullary_op(self);
      log_normal_kernel<RNG>()(iter, mean, std, gen);
      return self;
        */
}

// =================================================== Geometric ======================================================

pub fn geometric_impl<geometric_kernel, RNG>(
        self_: &mut Tensor,
        p:     f64,
        gen:   Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            TORCH_CHECK(0 < p && p < 1, "geometric_ expects p to be in (0, 1), but got p=", p);
      auto iter = TensorIterator::borrowing_nullary_op(self);
      geometric_kernel<RNG>()(iter, p, gen);
      return self;
        */
}

// ================================================== Exponential =====================================================

pub fn exponential_impl<exponential_kernel, RNG>(
        self_:  &mut Tensor,
        lambda: f64,
        gen:    Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            TORCH_CHECK(lambda >= 0.0, "exponential_ expects lambda >= 0.0, but found lambda=", lambda);
      auto iter = TensorIterator::borrowing_nullary_op(self);
      exponential_kernel<RNG>()(iter, lambda, gen);
      return self;
        */
}

// ==================================================== Cauchy ========================================================

pub fn cauchy_impl<cauchy_kernel, RNG>(
        self_:  &mut Tensor,
        median: f64,
        sigma:  f64,
        gen:    Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            auto iter = TensorIterator::borrowing_nullary_op(self);
      cauchy_kernel<RNG>()(iter, median, sigma, gen);
      return self;
        */
}

// ==================================================== Bernoulli =====================================================

pub fn bernoulli_impl_a<bernoulli_tensor_kernel, RNG>(
        self_: &mut Tensor,
        p:     &Tensor,
        gen:   Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            NoNamesGuard guard;
      assert_no_internal_overlap(self);
      bernoulli_tensor_kernel<RNG>()(self, p_, gen);
      return self;
        */
}

pub fn bernoulli_impl_b<bernoulli_scalar_kernel, RNG>(
        self_: &mut Tensor,
        p:     f64,
        gen:   Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            TORCH_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
      assert_no_internal_overlap(self);
      bernoulli_scalar_kernel<RNG>()(self, p, gen);
      return self;
        */
}

pub fn bernoulli_out_impl<bernoulli_tensor_kernel, RNG>(
        result: &mut Tensor,
        self_:  &Tensor,
        gen:    Option<Generator>) -> &mut Tensor {

    todo!();
        /*
            // result.resize_as_(self) requires self to have same dtype as result, so we
      // use resize_ instead.
      // TODO: Fix resize_as_. See pytorch/pytorch#11665.
      result.resize_(self.sizes());
      bernoulli_impl_<bernoulli_tensor_kernel, RNG>(result, self, gen);
      namedinference::propagate_names(result, self);
      return result;
        */
}
