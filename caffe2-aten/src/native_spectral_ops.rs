crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/SpectralOps.cpp]

/**
  | Promote inputs to FFT functions
  |
  | * Integers are promoted to the default floating
  | type
  |
  | * If require_complex=True, all types are
  | promoted to complex
  |
  | * Raises an error for half-precision dtypes to
  | allow future support
  |
  */
pub fn promote_type_fft(
    ty:              ScalarType,
    require_complex: bool) -> ScalarType {

    todo!();
        /*
            if (isComplexType(type)) {
        return type;
      }
      // Promote integral to default float type
      if (!isFloatingType(type)) {
        type = typeMetaToScalarType(get_default_dtype());
      }

      TORCH_CHECK(type == kFloat || type == kDouble, "Unsupported dtype ", type);

      if (!require_complex) {
        return type;
      }

      // Promote to complex
      switch (type) {
      case kFloat: return kComplexFloat;
      case kDouble: return kComplexDouble;
      default: TORCH_INTERNAL_ASSERT(false, "Unhandled dtype");
      }
        */
}

/**
  | Promote a tensor's dtype according
  | to promote_type_fft
  |
  */
pub fn promote_tensor_fft(
        t:               &Tensor,
        require_complex: bool) -> Tensor {
    let require_complex: bool = require_complex.unwrap_or(false);

    todo!();
        /*
            auto cur_type = t.scalar_type();
      auto new_type = promote_type_fft(cur_type, require_complex);
      return (cur_type == new_type) ? t : t.to(new_type);
        */
}

/**
  | Convert NumPy compatible normalization mode
  | string to enum values
  |
  | NOTE: NumPy's normalization modes have
  | direction-specific meanings. For example,
  | "forward" translates to `by_n` for a forward
  | transform and `none` for backward.
  |
  */
pub fn norm_from_string(
    norm:    Option<StringView>,
    forward: bool) -> FftNormMode {
    
    todo!();
        /*
            if (!norm || *norm == "backward") {
        return forward ? fft_norm_mode::none : fft_norm_mode::by_n;
      }

      if (*norm == "forward") {
        return forward ? fft_norm_mode::by_n : fft_norm_mode::none;
      }

      if (*norm == "ortho") {
        return fft_norm_mode::by_root_n;
      }

      TORCH_CHECK(false, "Invalid normalization mode: \"", *norm, "\"")
        */
}

/**
  | Fixes the shape of x such that
  | 
  | x.size(dims[i]) == sizes[i],
  | 
  | either by zero-padding, or by slicing
  | x starting from 0.
  |
  */
pub fn resize_fft_input(
    x:     Tensor,
    dims:  &[i32],
    sizes: &[i32]) -> Tensor {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(dims.size() == sizes.size());
      bool must_copy = false;
      auto x_sizes = x.sizes();
      DimVector pad_amount(x_sizes.size() * 2);
      for (i64 i = 0; i < dims.size(); ++i) {
        if (sizes[i] == -1) {
          continue;
        }

        if (x_sizes[dims[i]] < sizes[i]) {
          must_copy = true;
          auto pad_idx = pad_amount.size() - 2 * dims[i] - 1;
          pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]];
        }

        if (x_sizes[dims[i]] > sizes[i]) {
          x = x.slice(dims[i], 0, sizes[i]);
        }
      }

      // Only call pad if necessary since pad copies the entire tensor
      return must_copy ? constant_pad_nd(x, pad_amount) : x;
        */
}

/**
  | Complex to real FFT
  |
  */
pub fn fft_c2r(
        function_name: StringView,
        out:           Tensor,
        input:         Tensor,
        n_opt:         Option<i64>,
        unwrapped_dim: i64,
        norm_str:      Option<StringView>,
        forward:       bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!out.defined() || out.is_floating_point(), function_name,
                  " expects a floating point output tensor, but got ", out.scalar_type());
      input = promote_tensor_fft(input, /*require_complex=*/true);
      const auto input_dim = input.dim();
      const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim);
      const auto n = n_opt.value_or(2*(input.sizes()[dim] - 1));
      TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
      if (n_opt) {
        input = resize_fft_input(input, dim, n/2 + 1);
      }
      const auto norm = norm_from_string(norm_str, forward);
      if (forward) {
        // FIXME: _fft does not support complex_output=false with inverse=false
        input = conj(input);
      }
      if (out.defined()) {
        return _fft_c2r_out(out, input, dim, static_cast<i64>(norm), n);
      } else {
        return _fft_c2r(input, dim, static_cast<i64>(norm), n);
      }
        */
}

/**
  | Real to complex FFT
  |
  */
pub fn fft_r2c(
        function_name: StringView,
        out:           Tensor,
        input:         Tensor,
        n_opt:         Option<i64>,
        unwrapped_dim: i64,
        norm_str:      Option<StringView>,
        forward:       bool,
        onesided:      bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!input.is_complex(), function_name,
                  " expects a real input tensor, but got ", input.scalar_type());
      TORCH_CHECK(!out.defined() || out.is_complex(), function_name,
                  " expects a complex output tensor, but got ", out.scalar_type());
      input = promote_tensor_fft(input);
      const auto input_dim = input.dim();
      const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim);
      const auto n = n_opt.value_or(input.sizes()[dim]);
      TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
      if (n_opt) {
        input = resize_fft_input(input, dim, n);
      }

      const auto norm = norm_from_string(norm_str, forward);

      Tensor ret;
      if (out.defined() && forward) {
        ret = _fft_r2c_out(out, input, dim, static_cast<i64>(norm), onesided);
      } else {
        ret = _fft_r2c(input, dim, static_cast<i64>(norm), onesided);
      }

      if (!forward) {
        // FIXME: _fft_r2c doesn't support native r2c IFFT
        return out.defined() ? conj_physical_out(out, ret) : conj(ret);
      } else {
        return ret;
      }
        */
}

/**
  | Complex to complex FFT
  |
  */
pub fn fft_c2c(
        function_name: StringView,
        out:           Tensor,
        input:         Tensor,
        n_opt:         Option<i64>,
        unwrapped_dim: i64,
        norm_str:      Option<StringView>,
        forward:       bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(input.is_complex(), function_name,
                  " expects a complex input tensor, but got ", input.scalar_type());
      const auto input_dim = input.dim();
      const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim);
      const auto n = n_opt.value_or(input.sizes()[dim]);
      TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
      if (n_opt) {
        input = resize_fft_input(input, dim, n);
      }
      const auto norm = norm_from_string(norm_str, forward);
      if (out.defined()) {
        TORCH_CHECK(out.is_complex(), function_name,
                    " expects a complex output tensor, but got ", out.scalar_type());
        return _fft_c2c_out(out, input, dim, static_cast<i64>(norm), forward);
      } else {
        return _fft_c2c(input, dim, static_cast<i64>(norm), forward);
      }
        */
}

/**
  | Dimensions to transform, and the signal
  | shape in those dimensions
  |
  */
pub struct ShapeAndDims {
    shape: DimVector,
    dim:   DimVector,
}

/**
  | Pre-process n-dimensional fft's `s` and `dim`
  | arguments.
  |
  | Wraps dimensions and applies defaulting
  | behavior.
  |
  | Also checks transform dims are unique and
  | transform shape is non-empty.
  |
  */
pub fn canonicalize_fft_shape_and_dim_args(
        input: Tensor,
        shape: Option<&[i32]>,
        dim:   Option<&[i32]>) -> ShapeAndDims {
    
    todo!();
        /*
            const i64 input_dim = input.dim();
      const IntArrayRef input_sizes = input.sizes();
      ShapeAndDims ret;

      if (dim) {
        ret.dim.resize(dim->size());
        copy(dim->begin(), dim->end(), ret.dim.begin());
        maybe_wrap_dims(ret.dim, input_dim);

        // Check dims are unique
        DimVector copy = ret.dim;
        sort(copy.begin(), copy.end());
        auto duplicate = adjacent_find(copy.begin(), copy.end());
        TORCH_CHECK(duplicate == copy.end(), "FFT dims must be unique");
      }

      if (shape) {
        // Has shape, may have dim
        TORCH_CHECK(!dim || dim->size() == shape->size(),
                    "When given, dim and shape arguments must have the same length");
        TORCH_CHECK(shape->size() <= input_dim,
                    "Got shape with ", shape->size(), " values but input tensor "
                    "only has ", input_dim, " dimensions.");
        const i64 transform_ndim = shape->size();
        // If shape is given, dims defaults to the last shape.size() dimensions
        if (!dim) {
          ret.dim.resize(transform_ndim);
          iota(ret.dim.begin(), ret.dim.end(), input_dim - transform_ndim);
        }

        // Translate shape of -1 to the default length
        ret.shape.resize(transform_ndim);
        for (i64 i = 0; i < transform_ndim; ++i) {
          const auto n = (*shape)[i];
          ret.shape[i] = n == -1 ? input_sizes[ret.dim[i]] : n;
        }
      } else if (!dim) {
        // No shape, no dim
        ret.dim.resize(input_dim);
        iota(ret.dim.begin(), ret.dim.end(), i64{0});
        ret.shape.resize(input_dim);
        copy(input_sizes.begin(), input_sizes.end(), ret.shape.begin());
      } else {
        // No shape, has dim
        ret.shape.resize(ret.dim.size());
        for (i64 i = 0; i < ret.dim.size(); ++i) {
          ret.shape[i] = input_sizes[ret.dim[i]];
        }
      }

      for (i64 i = 0; i < ret.shape.size(); ++i) {
        TORCH_CHECK(ret.shape[i] > 0,
                    "Invalid number of data points (", ret.shape[i], ") specified");
      }

      return ret;
        */
}

/**
  | Complex to complex n-dimensional fft
  |
  */
pub fn fftn_c2c(
        function_name: StringView,
        out:           Tensor,
        input:         &Tensor,
        shape:         &[i32],
        dim:           &[i32],
        norm_str:      Option<StringView>,
        forward:       bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(input.is_complex(), function_name, " expects a complex input tensor, but got", input.scalar_type());
      Tensor x = resize_fft_input(input, dim, shape);
      const auto norm = norm_from_string(norm_str, forward);
      if (out.defined()) {
        TORCH_CHECK(out.is_complex(), function_name, " expects a complex output tensor, but got ", out.scalar_type());
        return _fft_c2c_out(out, x, dim, static_cast<i64>(norm), forward);
      } else {
        return _fft_c2c(x, dim, static_cast<i64>(norm), forward);
      }
        */
}

/**
  | torch.fft.fft, analogous to NumPy's
  | numpy.fft.fft
  |
  */
pub fn fft_fft(
    self_: &Tensor,
    n:     Option<i64>,
    dim:   i64,
    norm:  Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return self.is_complex() ?
        fft_c2c("fft", {}, self, n, dim, norm, /*forward=*/true) :
        fft_r2c("fft", {}, self, n, dim, norm, /*forward=*/true, /*onesided=*/false);
        */
}

pub fn fft_fft_out(
    self_: &Tensor,
    n:     Option<i64>,
    dim:   i64,
    norm:  Option<StringView>,
    out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            if (self.is_complex()) {
        fft_c2c("fft", out, self, n, dim, norm, /*forward=*/true);
      } else {
        fft_r2c("fft", out, self, n, dim, norm, /*forward=*/true, /*onesided=*/false);
      }
      return out;
        */
}


pub fn fft_ifft(
        self_: &Tensor,
        n:     Option<i64>,
        dim:   i64,
        norm:  Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return self.is_complex() ?
        fft_c2c("ifft", {}, self, n, dim, norm, /*forward=*/false) :
        fft_r2c("ifft", {}, self, n, dim, norm, /*forward=*/false, /*onesided=*/false);
        */
}


pub fn fft_ifft_out(
        self_: &Tensor,
        n:     Option<i64>,
        dim:   i64,
        norm:  Option<StringView>,
        out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            if (self.is_complex()) {
        fft_c2c("ifft", out, self, n, dim, norm, /*forward=*/false);
      } else {
        fft_r2c("ifft", out, self, n, dim, norm, /*forward=*/false, /*onesided=*/false);
      }
      return out;
        */
}


pub fn fft_rfft(
        self_: &Tensor,
        n:     Option<i64>,
        dim:   i64,
        norm:  Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return fft_r2c("rfft", {}, self, n, dim, norm, /*forward=*/true, /*onesided=*/true);
        */
}


pub fn fft_rfft_out(
        self_: &Tensor,
        n:     Option<i64>,
        dim:   i64,
        norm:  Option<StringView>,
        out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            fft_r2c("rfft", out, self, n, dim, norm, /*forward=*/true, /*onesided=*/true);
      return out;
        */
}


pub fn fft_irfft(
        self_: &Tensor,
        n:     Option<i64>,
        dim:   i64,
        norm:  Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return fft_c2r("irfft", {}, self, n, dim, norm, /*forward=*/false);
        */
}


pub fn fft_irfft_out(
        self_: &Tensor,
        n:     Option<i64>,
        dim:   i64,
        norm:  Option<StringView>,
        out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            fft_c2r("irfft", out, self, n, dim, norm, /*forward=*/false);
      return out;
        */
}


pub fn fft_hfft(
        self_: &Tensor,
        n:     Option<i64>,
        dim:   i64,
        norm:  Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return fft_c2r("hfft", {}, self, n, dim, norm, /*forward=*/true);
        */
}


pub fn fft_hfft_out(
        self_: &Tensor,
        n:     Option<i64>,
        dim:   i64,
        norm:  Option<StringView>,
        out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            fft_c2r("hfft", out, self, n, dim, norm, /*forward=*/true);
      return out;
        */
}


pub fn fft_ihfft(
        self_: &Tensor,
        n:     Option<i64>,
        dim:   i64,
        norm:  Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return fft_r2c("ihfft", {}, self, n, dim, norm, /*forward=*/false, /*onesided=*/true);
        */
}


pub fn fft_ihfft_out(
        self_: &Tensor,
        n:     Option<i64>,
        dim:   i64,
        norm:  Option<StringView>,
        out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            fft_r2c("ihfft", out, self, n, dim, norm, /*forward=*/false, /*onesided=*/true);
      return out;
        */
}


pub fn fft_fftn(
        self_: &Tensor,
        s:     Option<&[i32]>,
        dim:   Option<&[i32]>,
        norm:  Option<StringView>) -> Tensor {
    
    todo!();
        /*
            auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
      // TODO: For real input, perform rfftn then mirror with conjugate symmetry
      Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
      return fftn_c2c("fftn", {}, input, desc.shape, desc.dim, norm, /*forward=*/true);
        */
}


pub fn fft_fftn_out(
        self_: &Tensor,
        s:     Option<&[i32]>,
        dim:   Option<&[i32]>,
        norm:  Option<StringView>,
        out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
      // TODO: For real input, perform rfftn then mirror with conjugate symmetry
      Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
      fftn_c2c("fftn", out, input, desc.shape, desc.dim, norm, /*forward=*/true);
      return out;
        */
}


pub fn fft_ifftn(
        self_: &Tensor,
        s:     Option<&[i32]>,
        dim:   Option<&[i32]>,
        norm:  Option<StringView>) -> Tensor {
    
    todo!();
        /*
            auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
      Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
      return fftn_c2c("ifftn", {}, input, desc.shape, desc.dim, norm, /*forward=*/false);
        */
}


pub fn fft_ifftn_out(
        self_: &Tensor,
        s:     Option<&[i32]>,
        dim:   Option<&[i32]>,
        norm:  Option<StringView>,
        out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
      Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
      fftn_c2c("ifftn", out, input, desc.shape, desc.dim, norm, /*forward=*/false);
      return out;
        */
}


pub fn fft_rfftn_impl(
        out:      Tensor,
        self_:    &Tensor,
        s:        Option<&[i32]>,
        dim:      Option<&[i32]>,
        norm_str: &Option<StringView>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_complex(), "rfftn expects a real-valued input tensor, but got ", self.scalar_type());
      auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
      TORCH_CHECK(desc.shape.size() > 0, "rfftn must transform at least one axis");
      Tensor input = promote_tensor_fft(self, /*require_complex=*/false);
      Tensor x = resize_fft_input(input, desc.dim, desc.shape);
      const auto norm = norm_from_string(norm_str, /*forward=*/true);
      if (out.defined()) {
        TORCH_CHECK(out.is_complex(), "rfftn expects a complex-valued output tensor, but got ", out.scalar_type());
        return _fft_r2c_out(out, x, desc.dim, static_cast<i64>(norm), /*onesided=*/true);
      } else {
        return _fft_r2c(x, desc.dim, static_cast<i64>(norm), /*onesided=*/true);
      }
        */
}


pub fn fft_rfftn(
        self_:    &Tensor,
        s:        Option<&[i32]>,
        dim:      Option<&[i32]>,
        norm_str: Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return fft_rfftn_impl({}, self, s, dim, norm_str);
        */
}


pub fn fft_rfftn_out(
        self_:    &Tensor,
        s:        Option<&[i32]>,
        dim:      Option<&[i32]>,
        norm_str: Option<StringView>,
        out:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            fft_rfftn_impl(out, self, s, dim, norm_str);
      return out;
        */
}


pub fn fft_irfftn_impl(
        out:      Tensor,
        self_:    &Tensor,
        s:        Option<&[i32]>,
        dim:      Option<&[i32]>,
        norm_str: &Option<StringView>) -> Tensor {
    
    todo!();
        /*
            auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
      TORCH_CHECK(desc.shape.size() > 0, "irfftn must transform at least one axis");

      const auto last_dim_size = [&] {
        // Fixup default shape handling in the last dimension,
        if (!s.has_value() || (s->back() == -1)) {
          const auto last_dim = desc.dim.back();
          return 2 * (self.sizes()[last_dim] - 1);
        }
        return desc.shape.back();
      }();
      desc.shape.back() = last_dim_size / 2 + 1;

      Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
      Tensor x = resize_fft_input(input, desc.dim, desc.shape);
      const auto norm = norm_from_string(norm_str, /*forward=*/false);
      if (out.defined()) {
        TORCH_CHECK(out.is_floating_point(), "irfftn expects a floating point output tensor, but got ", out.scalar_type());
        return _fft_c2r_out(out, x, desc.dim, static_cast<i64>(norm), last_dim_size);
      } else {
        return _fft_c2r(x, desc.dim, static_cast<i64>(norm), last_dim_size);
      }
        */
}


pub fn fft_irfftn(
        self_:    &Tensor,
        s:        Option<&[i32]>,
        dim:      Option<&[i32]>,
        norm_str: Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return fft_irfftn_impl({}, self, s, dim, norm_str);
        */
}


pub fn fft_irfftn_out(
        self_:    &Tensor,
        s:        Option<&[i32]>,
        dim:      Option<&[i32]>,
        norm_str: Option<StringView>,
        out:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            fft_irfftn_impl(out, self, s, dim, norm_str);
      return out;
        */
}


pub fn fft_fft2(
        self_: &Tensor,
        s:     Option<&[i32]>,
        dim:   &[i32],
        norm:  Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return native::fft_fftn(self, s, dim, move(norm));
        */
}


pub fn fft_fft2_out(
        self_: &Tensor,
        s:     Option<&[i32]>,
        dim:   &[i32],
        norm:  Option<StringView>,
        out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return native::fft_fftn_out(self, s, dim, move(norm), out);
        */
}


pub fn fft_ifft2(
        self_: &Tensor,
        s:     Option<&[i32]>,
        dim:   &[i32],
        norm:  Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return native::fft_ifftn(self, s, dim, move(norm));
        */
}


pub fn fft_ifft2_out(
        self_: &Tensor,
        s:     Option<&[i32]>,
        dim:   &[i32],
        norm:  Option<StringView>,
        out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return native::fft_ifftn_out(self, s, dim, move(norm), out);
        */
}


pub fn fft_rfft2(
        self_: &Tensor,
        s:     Option<&[i32]>,
        dim:   &[i32],
        norm:  Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return native::fft_rfftn(self, s, dim, move(norm));
        */
}


pub fn fft_rfft2_out(
        self_: &Tensor,
        s:     Option<&[i32]>,
        dim:   &[i32],
        norm:  Option<StringView>,
        out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return native::fft_rfftn_out(self, s, dim, move(norm), out);
        */
}


pub fn fft_irfft2(
        self_: &Tensor,
        s:     Option<&[i32]>,
        dim:   &[i32],
        norm:  Option<StringView>) -> Tensor {
    
    todo!();
        /*
            return native::fft_irfftn(self, s, dim, move(norm));
        */
}


pub fn fft_irfft2_out(
        self_: &Tensor,
        s:     Option<&[i32]>,
        dim:   &[i32],
        norm:  Option<StringView>,
        out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return native::fft_irfftn_out(self, s, dim, move(norm), out);
        */
}


pub fn fft_fftfreq_out(
        n:   i64,
        d:   f64,
        out: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            ScalarType dtype = out.scalar_type();
      TORCH_CHECK(isFloatingType(dtype) || isComplexType(dtype),
                  "fftfreq requires a floating point or complex dtype");
      // TODO: arange doesn't have complex support
      arange_out(out, n);
      auto right_slice = out.slice(0, (n + 1) / 2, 0);
      arange_out(right_slice, -(n/2), 0, 1);
      return out.mul_(1.0 / (n * d));  // Slightly faster than div_(n*d)
        */
}


pub fn fft_fftfreq(
        n:          i64,
        d:          f64,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto out = empty({n}, options);
      return native::fft_fftfreq_out(n, d, out);
        */
}


pub fn fft_rfftfreq_out(
        n:   i64,
        d:   f64,
        out: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            ScalarType dtype = out.scalar_type();
      TORCH_CHECK(isFloatingType(dtype) || isComplexType(dtype),
                  "rfftfreq requires a floating point or complex dtype");
      // TODO: arange doesn't have complex support
      native::arange_out(n/2 + 1, out);
      return out.mul_(1.0 / (n * d));  // Slightly faster than div_(n*d)
        */
}


pub fn fft_rfftfreq(
        n:          i64,
        d:          f64,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto out = empty({n/2 + 1}, options);
      return native::fft_rfftfreq_out(n, d, out);
        */
}

/**
  | If an array dim is specified, wraps them
  | according to self.dim().
  | 
  | Otherwise returns a vector of all dims.
  |
  */
pub fn default_alldims(
        self_:   &Tensor,
        dim_opt: Option<&[i32]>) -> DimVector {
    
    todo!();
        /*
            DimVector dim;
      if (dim_opt) {
        IntArrayRef dim_unwrapped = *dim_opt;
        dim.resize(dim_unwrapped.size());
        for (i64 i = 0; i < dim.size(); ++i) {
          dim[i] = maybe_wrap_dim(dim_unwrapped[i], self.dim());
        }
      } else {
        dim.resize(self.dim());
        iota(dim.begin(), dim.end(), 0);
      }
      return dim;
        */
}



pub fn fft_fftshift(
        x:       &Tensor,
        dim_opt: Option<&[i32]>) -> Tensor {
    
    todo!();
        /*
            auto dim = default_alldims(x, dim_opt);

      IntArrayRef x_sizes = x.sizes();
      DimVector shift(dim.size());
      for (i64 i = 0; i < dim.size(); ++i) {
        shift[i] = x_sizes[dim[i]] / 2;
      }

      return roll(x, shift, dim);
        */
}


pub fn fft_ifftshift(
        x:       &Tensor,
        dim_opt: Option<&[i32]>) -> Tensor {
    
    todo!();
        /*
            auto dim = default_alldims(x, dim_opt);

      IntArrayRef x_sizes = x.sizes();
      DimVector shift(dim.size());
      for (i64 i = 0; i < dim.size(); ++i) {
        shift[i] = (x_sizes[dim[i]] + 1) / 2;
      }

      return roll(x, shift, dim);
        */
}

/**
  | We call the following methods via CUDA
  | hooks because they are really only valid
  | when CUDA is available. See native/cuda/CuFFTPlanCache.h
  | for more details.
  |
  */
pub fn cufft_get_plan_cache_max_size(device_index: i64) -> i64 {
    
    todo!();
        /*
            return getCUDAHooks().cuFFTGetPlanCacheMaxSize(device_index);
        */
}


pub fn cufft_set_plan_cache_max_size(
        device_index: i64,
        max_size:     i64)  {
    
    todo!();
        /*
            getCUDAHooks().cuFFTSetPlanCacheMaxSize(device_index, max_size);
        */
}


pub fn cufft_get_plan_cache_size(device_index: i64) -> i64 {
    
    todo!();
        /*
            return getCUDAHooks().cuFFTGetPlanCacheSize(device_index);
        */
}


pub fn cufft_clear_plan_cache(device_index: i64)  {
    
    todo!();
        /*
            getCUDAHooks().cuFFTClearPlanCache(device_index);
        */
}

pub fn write_opt<Stream, T>(
        SS:    &mut Stream,
        value: &Option<T>) -> &mut Stream {

    todo!();
        /*
            if (value) {
        SS << *value;
      } else {
        SS << "None";
      }
      return SS;
        */
}


/**
  | Short-time Fourier Transform, for
  | signal analysis.
  | 
  | This is modeled after librosa but with
  | support for complex time-domain signals
  | and complex windows.
  | 
  | -----------
  | @note
  | 
  | librosa's center and pad_mode arguments
  | are currently only implemented in python
  | because it uses torch.nn.functional.pad
  | which is python-only.
  |
  */
pub fn stft_a(
        self_:              &Tensor,
        n_fft:              i64,
        hop_length_opt:     Option<i64>,
        win_length_opt:     Option<i64>,
        window_opt:         &Option<Tensor>,
        normalized:         bool,
        onesided_opt:       Option<bool>,
        return_complex_opt: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> window_maybe_owned = borrow_from_optional_tensor(window_opt);
      const Tensor& window = *window_maybe_owned;

      #define REPR(SS) \
        SS << "stft(" << self.toString() << self.sizes() << ", n_fft=" << n_fft \
           << ", hop_length=" << hop_length << ", win_length=" << win_length \
           << ", window="; \
        if (window.defined()) { \
          SS << window.toString() << "{" << window.sizes() << "}"; \
        } else { \
          SS << "None"; \
        } \
        SS << ", normalized=" << normalized << ", onesided="; \
        write_opt(SS, onesidedOpt) << ", return_complex="; \
        write_opt(SS, return_complexOpt) << ") "

      TORCH_CHECK(!window.defined() || window.device() == self.device(),
                  "stft input and window must be on the same device but got self on ",
                  self.device(), " and window on ", window.device())

      // default_init hop_length and win_length
      auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
      auto win_length = win_lengthOpt.value_or(n_fft);
      const bool return_complex = return_complexOpt.value_or(
          self.is_complex() || (window.defined() && window.is_complex()));
      if (!return_complex) {
        if (!return_complexOpt.has_value()) {
          TORCH_WARN_ONCE(
            "stft will soon require the return_complex parameter be given for real inputs, "
            "and will further require that return_complex=True in a future PyTorch release."
          );
        }

        // TORCH_WARN_ONCE(
        //     "stft with return_complex=False is deprecated. In a future pytorch "
        //     "release, stft will return complex tensors for all inputs, and "
        //     "return_complex=False will raise an error.\n"
        //     "Note: you can still call torch.view_as_real on the complex output to "
        //     "recover the old return format.");
      }

      if (!isFloatingType(self.scalar_type()) && !isComplexType(self.scalar_type())) {
        ostringstream ss;
        REPR(ss) << ": expected a tensor of floating point or complex values";
        AT_ERROR(ss.str());
      }
      if (self.dim() > 2 || self.dim() < 1) {
        ostringstream ss;
        REPR(ss) << ": expected a 1D or 2D tensor";
        AT_ERROR(ss.str());
      }
      Tensor input = self;
      if (self.dim() == 1) {
        input = input.unsqueeze(0);
      }
      i64 batch = input.size(0);
      i64 len = input.size(1);
      if (n_fft <= 0 || n_fft > len) {
        ostringstream ss;
        REPR(ss) << ": expected 0 < n_fft < " << len
                 << ", but got n_fft=" << win_length;
        AT_ERROR(ss.str());
      }
      if (hop_length <= 0) {
        ostringstream ss;
        REPR(ss) << ": expected hop_length > 0, but got hop_length=" << hop_length;
        AT_ERROR(ss.str());
      }
      if (win_length <= 0 || win_length > n_fft) {
        ostringstream ss;
        REPR(ss) << ": expected 0 < win_length <= n_fft, but got win_length="
                 << win_length;
        AT_ERROR(ss.str());
      }
      if (window.defined() && (window.dim() != 1 || window.size(0) != win_length)) {
        ostringstream ss;
        REPR(ss) << ": expected a 1D window tensor of size equal to win_length="
                 << win_length << ", but got window with size " << window.sizes();
        AT_ERROR(ss.str());
      }
      #undef REPR
      auto window_ = window;
      if (win_length < n_fft) {
        // pad center
        auto left = (n_fft - win_length) / 2;
        if (window.defined()) {
          window_ = zeros({n_fft}, window.options());
          window_.narrow(0, left, win_length).copy_(window);
        } else {
          window_ = zeros({n_fft}, self.options());
          window_.narrow(0, left, win_length).fill_(1);
        }
      }
      i64 n_frames = 1 + (len - n_fft) / hop_length;
      // time2col
      input = input.as_strided(
        {batch, n_frames, n_fft},
        {input.stride(0), hop_length * input.stride(1), input.stride(1)}
      );
      if (window_.defined()) {
        input = input.mul(window_);
      }

      // FFT and transpose to get (batch x fft_size x num_frames)
      const bool complex_fft = input.is_complex();
      const auto onesided = onesidedOpt.value_or(!complex_fft);

      const fft_norm_mode norm = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none;
      Tensor out;
      if (complex_fft) {
        TORCH_CHECK(!onesided, "Cannot have onesided output if window or input is complex");
        out = _fft_c2c(input, input.dim() - 1, static_cast<i64>(norm), /*forward=*/true);
      } else {
        out = _fft_r2c(input, input.dim() - 1, static_cast<i64>(norm), onesided);
      }
      out.transpose_(1, 2);

      if (self.dim() == 1) {
        out.squeeze_(0);
      }

      if (return_complex) {
        return out;
      } else {
        return view_as_real(out);
      }
        */
}

/**
  | Create complex tensor from the old style
  | of real tensor with size=(..., 2)
  | 
  | This is to support istft in the transition
  | to requiring complex input.
  | 
  | -----------
  | @note
  | 
  | This may return a view of the input tensor,
  | or might clone if necessary
  |
  */

pub fn as_complex(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            const bool can_view_as_complex = [&]{
        auto strides = self.strides();
        for (i64 i = 0; i + 1 < strides.size(); ++i) {
          if (strides[i] % 2 != 0) {
            return false;
          }
        }
        return strides.back() == 1 && self.storage_offset() % 2 == 0;
      }();
      return view_as_complex(can_view_as_complex ? self : self.clone(MemoryFormat::Contiguous));
        */
}


/**
  | Inverse Short-time Fourier Transform
  | 
  | This is modeled after librosa but with
  | support for complex time-domain signals
  | and complex windows.
  |
  */
pub fn istft_a(
    self_:          &Tensor,
    n_fft:          i64,
    hop_length_opt: Option<i64>,
    win_length_opt: Option<i64>,
    window_opt:     &Option<Tensor>,
    center:         bool,
    normalized:     bool,
    onesided_opt:   Option<bool>,
    length_opt:     Option<i64>,
    return_complex: bool) -> Tensor {

    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> window_maybe_owned = borrow_from_optional_tensor(window_opt);
      const Tensor& window = *window_maybe_owned;

      #define REPR(SS) \
        SS << "istft(" << self.toString() << self.sizes() << ", n_fft=" << n_fft \
           << ", hop_length=" << hop_length << ", win_length=" << win_length \
           << ", window="; \
        if (window.defined()) { \
          SS << window.toString() << "{" << window.sizes() << "}"; \
        } else { \
          SS << "None"; \
        } \
        SS << ", center=" << center << ", normalized=" << normalized << ", onesided="; \
        write_opt(SS, onesidedOpt) << ", length="; \
        write_opt(SS, lengthOpt) << ", return_complex=" << return_complex << ") "

      TORCH_CHECK(!window.defined() || window.device() == self.device(),
                  "istft input and window must be on the same device but got self on ",
                  self.device(), " and window on ", window.device())

      // default_init hop_length and win_length
      const auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
      const auto win_length = win_lengthOpt.value_or(n_fft);

      if (!self.is_complex()) {
        TORCH_WARN_ONCE(
          "istft will require a complex-valued input tensor in a future PyTorch release. "
          "Matching the output from stft with return_complex=True. ");
      }
      Tensor input = self.is_complex() ? view_as_real(self) : self;
      const auto input_dim = input.dim();
      const auto n_frames = input.size(-2);
      const auto fft_size = input.size(-3);

      const auto expected_output_signal_len = n_fft + hop_length * (n_frames - 1);

      const auto options = device(input.device()).dtype(input.dtype());
      if (input.numel() == 0) {
        ostringstream ss;
        REPR(ss) << ": input tensor cannot be empty.";
        AT_ERROR(ss.str());
      }
      if (input_dim != 3 && input_dim != 4) {
        ostringstream ss;
        REPR(ss) << ": expected a tensor with 3 or 4 dimensions, but got " << input_dim;
        AT_ERROR(ss.str());
      }
      if (input.size(-1) != 2) {
        ostringstream ss;
        REPR(ss) << ": expected the last dimension to be 2 (corresponding to real and imaginary parts), but got " << self.size(-1);
        AT_ERROR(ss.str());
      }

      const bool onesided = onesidedOpt.value_or(fft_size != n_fft);
      if (onesided) {
        if (n_fft / 2 + 1 != fft_size) {
          ostringstream ss;
          REPR(ss) << ": expected the frequency dimension (3rd to the last) of the input tensor to match n_fft / 2 + 1 when onsided=True, but got " << fft_size;
          AT_ERROR(ss.str());
        }
      } else {
        if (n_fft != fft_size) {
          ostringstream ss;
          REPR(ss) << ": expected the frequency dimension (3rd to the last) of the input tensor to match n_fft when onsided=False, but got " << fft_size;
          AT_ERROR(ss.str());
        }
      }

      if (!(0 < hop_length && hop_length <= win_length)) {
        ostringstream ss;
        REPR(ss) << ": expected 0 < hop_length <= win_length";
        AT_ERROR(ss.str());
      }

      if (!(0 < win_length && win_length <= n_fft)) {
        ostringstream ss;
        REPR(ss) << ": expected 0 < win_length <= n_fft";
        AT_ERROR(ss.str());
      }
      if (window.defined()) {
        if (window.dim() != 1 || window.size(0) != win_length) {
          ostringstream ss;
          REPR(ss) << ": Invalid window shape. window has to be 1D and length of `win_length`";
          AT_ERROR(ss.str());
        }
      }

      Tensor window_tmp = window.defined() ? window : ones({win_length,}, options);
      if (win_length != n_fft) {
        // center window by padding zeros on right and left side
        i64 left = (n_fft - win_length) / 2;
        window_tmp = constant_pad_nd(window_tmp, {left, n_fft - win_length - left}, 0);
        TORCH_INTERNAL_ASSERT(window_tmp.size(0) == n_fft);
      }

      if (input_dim == 3) {
        input = input.unsqueeze(0);
      }

      input = as_complex(input.transpose(1, 2));  // size: (channel, n_frames, fft_size, 2)

      const fft_norm_mode norm = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::by_n;
      if (return_complex) {
        TORCH_CHECK(!onesided, "Cannot have onesided output if window or input is complex");
        input = _fft_c2c(input, input.dim() - 1, static_cast<i64>(norm), /*forward=*/false);  // size: (channel, n_frames, n_fft)
      } else {
        TORCH_CHECK(!window.defined() || !window.is_complex(),
                    "Complex windows are incompatible with return_complex=False");
        if (!onesided) {
          input = input.slice(-1, 0, n_fft / 2 + 1);
        }
        input = _fft_c2r(input, input.dim() - 1, static_cast<i64>(norm), n_fft);  // size: (channel, n_frames, n_fft)
      }
      TORCH_INTERNAL_ASSERT(input.size(2) == n_fft);

      Tensor y_tmp = input * window_tmp.view({1, 1, n_fft});  // size: (channel, n_frames, n_fft)
      y_tmp = y_tmp.transpose(1, 2);  // size: (channel, n_fft, frame)

      Tensor y = col2im(y_tmp,
                                      /*output_size*/ {1, (n_frames - 1) * hop_length + n_fft},
                                      /*kernel_size*/ {1, n_fft},
                                      /*dilation*/    {1, 1},
                                      /*padding*/     {0, 0},
                                      /*stride*/      {1, hop_length}
                                     ).squeeze(2);
      window_tmp = window_tmp.pow(2).view({n_fft, 1}).repeat({1, n_frames}).unsqueeze(0);  // size: (1, n_fft, n_frames)
      Tensor window_envelop = col2im(window_tmp,
                                      /*output_size*/ {1, (n_frames - 1) * hop_length + n_fft},
                                      /*kernel_size*/ {1, n_fft},
                                      /*dilation*/    {1, 1},
                                      /*padding*/     {0, 0},
                                      /*stride*/      {1, hop_length}
                                     ).squeeze(2); // size: (1, 1, expected_output_signal_len)

      TORCH_INTERNAL_ASSERT(expected_output_signal_len == y.size(2));
      TORCH_INTERNAL_ASSERT(expected_output_signal_len == window_envelop.size(2));

      // We need to trim the front padding away if centered
      const auto start = center ? n_fft / 2 : 0;
      const auto end = lengthOpt.has_value()? start + lengthOpt.value() : - n_fft / 2;

      y = y.slice(2, start, end, 1);
      window_envelop = window_envelop.slice(2, start, end, 1);
      const auto window_envelop_lowest = window_envelop.abs().min().item().toDouble();
      if (window_envelop_lowest < 1e-11) {
        ostringstream ss;
        REPR(ss) << "window overlap add min: " << window_envelop_lowest;
        AT_ERROR(ss.str());
      }

      y = (y / window_envelop).squeeze(1);  // size: (channel, expected_output_signal_len)
      if (input_dim == 3) {
        y = y.squeeze(0);
      }
      return y;

      #undef REPR
        */
}

pub fn stft_b(
    self_:          &Tensor,
    n_fft:          i64,
    hop_length_opt: Option<i64>,
    win_length_opt: Option<i64>,
    window:         &Tensor,
    normalized:     bool,
    onesided_opt:   Option<bool>) -> Tensor {

    todo!();
        /*
            return native::stft(
          self, n_fft, hop_lengthOpt, win_lengthOpt, window, normalized, onesidedOpt,
          /*return_complex=*/nullopt);
        */
}


pub fn istft_b(
        self_:          &Tensor,
        n_fft:          i64,
        hop_length_opt: Option<i64>,
        win_length_opt: Option<i64>,
        window:         &Tensor,
        center:         bool,
        normalized:     bool,
        onesided_opt:   Option<bool>,
        length_opt:     Option<i64>) -> Tensor {
    
    todo!();
        /*
            return native::istft(
          self, n_fft, hop_lengthOpt, win_lengthOpt, window, center, normalized,
          onesidedOpt, lengthOpt, /*return_complex=*/false);
        */
}


pub fn fft_fill_with_conjugate_symmetry(
        input: &Tensor,
        dim:   &[i32])  {
    
    todo!();
        /*
            const auto input_sizes = input.sizes();
      const auto input_strides = input.strides();
      TORCH_CHECK(dim_.size() > 0);
      DimVector dim(dim_.begin(), dim_.end());
      maybe_wrap_dims(dim, input_strides.size());

      if (input.numel() == 0 || input_sizes[dim.back()] <= 2) {
        return;  // No elements need writing
      }

      // Small dimensions may be treated as batch dims since they don't get mirrored
      dim.erase(
          remove_if(dim.begin(), dim.end(), [&](i64 dim) {
            return (input_sizes[dim] <= 2);
          }),
          dim.end());

      // Use TensorIterator to coalesce batch dimensions
      // NOTE: Can't use TensorIterator loops because we need negative strides
      auto iter = TensorIteratorConfig()
          .add_output(input)
          .add_input(input)
          .resize_outputs(false)
          .declare_static_shape(input_sizes, dim)
          .build();

      const auto iter_strides = iter.strides(0);
      const auto iter_sizes = iter.shape();
      const auto ndim = iter_strides.size() + dim.size();
      DimVector in_strides(ndim), signal_half_sizes(ndim);
      // Take coalesced batch dimensions from TensorIterator
      copy(iter_strides.begin(), iter_strides.end(), in_strides.begin());
      copy(iter_sizes.begin(), iter_sizes.end(), signal_half_sizes.begin());

      // Take transformed dimensions directly from the input
      const auto element_size = iter.element_size(0);
      for (i64 i = 0; i < dim.size(); ++i) {
        // Convert to byte strides to match TensorIterator
        in_strides[iter_strides.size() + i] = input_strides[dim[i]] * element_size;
        signal_half_sizes[iter_strides.size() + i] = input_sizes[dim[i]];
      }

      // For the last dimension, use negative strides to perform the mirroring
      signal_half_sizes.back() = (input_sizes[dim.back()] - 1) / 2;
      auto out_strides = in_strides;
      out_strides.back() *= -1;

      auto* data_ptr = static_cast<char*>(input.data_ptr());
      const auto* in_data = data_ptr + input_strides[dim.back()] * element_size;
      auto* out_data = data_ptr + (
          input_strides[dim.back()] * (input_sizes[dim.back()] - 1) * element_size);

      // Reorder dimensions by stride to maximize data locality
      DimVector dim_permute(ndim);
      iota(dim_permute.begin(), dim_permute.end(), 0);
      sort(dim_permute.begin(), dim_permute.end(),
          [&](auto dim1, auto dim2) {
            return in_strides[dim1] < in_strides[dim2];
          });

      DimVector temp(ndim);
      auto apply_permutation = [&] (DimVector & vec) {
        // Do permuted index copy into a temporary, then copy back
        for (i64 i = 0; i < ndim; ++i) {
          temp[i] = vec[dim_permute[i]];
        }
        vec = temp;
      };
      apply_permutation(in_strides);
      apply_permutation(out_strides);
      apply_permutation(signal_half_sizes);

      // Find dims.slice(dims.size() - 1) in the new permuted order.
      // These are the dimensions that need explicit Hermitian mirroring
      DimVector mirror_dims;
      mirror_dims.reserve(dim.size() - 1);
      for (i64 i = 0; i < ndim; ++i) {
        if (dim_permute[i] >= iter_strides.size() &&  // Not a batch dimension
            dim_permute[i] != ndim - 1) {  // Not the last dim, which is mirrored separately with negative strides
          mirror_dims.push_back(i);
        }
      }
      TORCH_INTERNAL_ASSERT(mirror_dims.size() == dim.size() - 1);

      // Dispatch to CPU or CUDA kernel to do the actual conjugate mirroring
      fft_fill_with_conjugate_symmetry_stub(
          input.device().type(), input.scalar_type(),
          mirror_dims, signal_half_sizes, in_strides, in_data, out_strides, out_data);
        */
}

define_dispatch!{fft_fill_with_conjugate_symmetry_stub}
