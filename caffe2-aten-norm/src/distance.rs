crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Distance.h]

lazy_static!{
    /*
    using pdist_forward_fn = void(*)(Tensor&, const Tensor&, const double p);
    using pdist_backward_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const double p, const Tensor&);
    using cdist_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const double p);
    using cdist_backward_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const Tensor&, const double p, const Tensor&);
    */
}

declare_dispatch!{pdist_forward_fn, pdist_forward_stub}
declare_dispatch!{pdist_backward_fn, pdist_backward_stub}
declare_dispatch!{cdist_fn, cdist_stub}
declare_dispatch!{cdist_backward_fn, cdist_backward_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Distance.cpp]

define_dispatch!{pdist_forward_stub}
define_dispatch!{pdist_backward_stub}
define_dispatch!{cdist_stub}       
define_dispatch!{cdist_backward_stub}

pub fn pairwise_distance(
        x1:      &Tensor,
        x2:      &Tensor,
        p:       f64,
        eps:     f64,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
            return norm(x1 - x2 + eps, p, 1, keepdim);
        */
}

/**
  | This is to guarantee that the contiguous
  | memory is passed to the backward pass
  |
  */
pub fn pdist(
        self_: &Tensor,
        p:     f64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.dim() == 2,
          "pdist only supports 2D tensors, got: ", self.dim(), "D");
      TORCH_CHECK(isFloatingType(self.scalar_type()), "pdist only supports floating-point dtypes");
      TORCH_CHECK(p >= 0, "pdist only supports non-negative p values");
      return _pdist_forward(self.contiguous(), p);
        */
}

pub fn euclidean_dist(
        x1: &Tensor,
        x2: &Tensor) -> Tensor {
    
    todo!();
        /*
            /** This function does the fist part of the euclidean distance calculation
       * We divide it in two steps to simplify dealing with subgradients in the
       * backward step */
      Tensor x1_norm = x1.pow(2).sum(-1, true);
      Tensor x1_pad = ones_like(x1_norm, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor x2_norm = x2.pow(2).sum(-1, true);
      Tensor x2_pad = ones_like(x2_norm, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor x1_ = cat({x1.mul(-2), x1_norm, x1_pad}, -1);
      Tensor x2_ = cat({x2, x2_pad, x2_norm}, -1);
      Tensor result = x1_.matmul(x2_.transpose(-2, -1));
      result.clamp_min_(0).sqrt_();
      return result;
        */
}

pub fn cdist_impl(
        x1:           &Tensor,
        x2:           &Tensor,
        p:            f64,
        compute_mode: Option<i64>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(isFloatingType(x1.scalar_type()), "cdist only supports floating-point dtypes, X1 got: ", x1.scalar_type());
      auto device1 = x1.device().type();
      TORCH_CHECK(isFloatingType(x1.scalar_type()), "cdist only supports floating-point dtypes, X2 got: ", x2.scalar_type());
      auto device2 = x2.device().type();
      TORCH_CHECK(p >= 0, "cdist only supports non-negative p values");
      TORCH_CHECK(device1 == device2, "X1 and X2 must have the same device type. X1: ", device1, " X2: ", device2);
      // TODO: This is bad; this test should apply universally
      TORCH_CHECK(!x1.is_cuda() || x1.get_device() == x2.get_device(), "device of X1 (", x1.get_device(), ") must match device of X2 (", x2.get_device(), ")");
      i64 c1 = x1.size(-1);
      i64 c2 = x2.size(-1);
      // 0 - default value. If p = 2 and r1 > 25 or r2 > 25 (these values are based on performance metrics),
      // it will try to compute distance using matrix multiplication approach
      // 1 - force to use matrix multiplication for p = 2
      // 2 - do not use matrix multiplication for p = 2
      i64 mode = compute_mode.value_or(0);
      TORCH_CHECK(mode >= 0 && mode <= 2, "possible modes: 0, 1, 2, but was: ", mode);

      i64 r1 = x1.size(-2);
      i64 r2 = x2.size(-2);

      // See Note [cdist relies on cdist_impl redispatching]
      // Keep this condition in sync with the condition at the Note
      if (!(p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25))))) {
        TORCH_CHECK(device1 == kCPU || device1 == kCUDA, "cdist only supports CPU and CUDA devices, X1 got: ", device1);
        TORCH_CHECK(device2 == kCPU || device2 == kCUDA, "cdist only supports CPU and CUDA devices, X2 got: ", device2);
      }

      auto dim1 = x1.dim();
      auto dim2 = x2.dim();

      //For batch calculation we expand all dimensions(except the last two) to one, with size that equals to product of them.
      //The last two dimensions will stay the same
      IntArrayRef batch_tensor1(x1.sizes().data(), dim1 - 2);
      IntArrayRef batch_tensor2(x2.sizes().data(), dim2 - 2);
      vector<i64> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);
      vector<i64> tensor1_expand_size(expand_batch_portion);
      tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
      vector<i64> tensor2_expand_size(expand_batch_portion);
      tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});

      const i64 expand_batch_product = multiply_integers(expand_batch_portion);
      vector<i64> tensor1_view{expand_batch_product, r1, c1};
      vector<i64> tensor2_view{expand_batch_product, r2, c2};

      Tensor tensor1_expanded = x1.expand(tensor1_expand_size).contiguous().view(tensor1_view);
      Tensor tensor2_expanded = x2.expand(tensor2_expand_size).contiguous().view(tensor2_view);

      vector<i64> output_shape(expand_batch_portion);
      output_shape.insert(output_shape.end(), {r1, r2});

      Tensor result;
      if (r1 == 0 || r2 == 0 || expand_batch_product == 0) {
        result = empty(output_shape, x1.options());
      } else if (c1 == 0) {
        result = zeros(output_shape, x1.options());
      } else if (p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25)))) {
        // See Note [cdist relies on cdist_impl redispatching]
        // Keep the condition above in sync with the condition at the Note
        Tensor dist = (expand_batch_product == 1) ? _euclidean_dist(x1, x2) :
                      _euclidean_dist(tensor1_expanded, tensor2_expanded);
        result = dist.view(output_shape);
      } else {
        result = empty(output_shape, x1.options());
        cdist_stub(device1, result, tensor1_expanded, tensor2_expanded, p);
      }
      return result;
        */
}

pub fn cdist(
        x1:           &Tensor,
        x2:           &Tensor,
        p:            f64,
        compute_mode: Option<i64>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(x1.dim() >= 2, "cdist only supports at least 2D tensors, X1 got: ", x1.dim(), "D");
      TORCH_CHECK(x2.dim() >= 2, "cdist only supports at least 2D tensors, X2 got: ", x2.dim(), "D");
      TORCH_CHECK(x1.size(-1) == x2.size(-1), "X1 and X2 must have the same number of columns. X1: ", x1.size(-1), " X2: ", x2.size(-1));
      auto maybe_outnames = namedinference::compute_cdist_outnames(x1, x2);
      auto result = [&]() {
        NoNamesGuard guard;
        i64 r1 = x1.size(-2);
        i64 r2 = x2.size(-2);
        // Special case for empty input: always call the version with explicit autograd to ensure the graph is properly connected
        if (x1.numel() == 0 || x2.numel() == 0) {
            return _cdist_forward(x1, x2, p, compute_mode);
        }
        i64 mode = compute_mode.value_or(0);
        // Note [cdist relies on cdist_impl redispatching]
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // This is for pytorch to figure the backward pass itself
        // when p=2.  Keep this condition in sync with the See Note reference
        if (p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25)))) {
            return cdist_impl(x1, x2, p, compute_mode);
        } else {
            return _cdist_forward(x1, x2, p, compute_mode);
        }
      }();
      namedinference::propagate_names_if_nonempty(result, maybe_outnames);
      return result;
        */
}

pub fn cdist_forward(
        x1:           &Tensor,
        x2:           &Tensor,
        p:            f64,
        compute_mode: Option<i64>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(x1.dim() >= 2, "cdist only supports at least 2D tensors, X1 got: ", x1.dim(), "D");
      TORCH_CHECK(x2.dim() >= 2, "cdist only supports at least 2D tensors, X2 got: ", x2.dim(), "D");
      TORCH_CHECK(x1.size(-1) == x2.size(-1), "X1 and X2 must have the same number of columns. X1: ", x1.size(-1), " X2: ", x2.size(-1));
      auto maybe_outnames = namedinference::compute_cdist_outnames(x1, x2);
      auto result = [&]() {
        NoNamesGuard guard;
        return cdist_impl(x1, x2, p, compute_mode);
      }();
      namedinference::propagate_names_if_nonempty(result, maybe_outnames);
      return result;
        */
}

pub fn cdist_backward(
        grad:  &Tensor,
        x1:    &Tensor,
        x2:    &Tensor,
        p:     f64,
        cdist: &Tensor) -> Tensor {
    
    todo!();
        /*
            // Broadcasting might generate non-contiguous Tensors, so handle it before doing checks
      i64 c1 = _x1.size(-1);
      i64 c2 = _x2.size(-1);
      i64 r1 = _x1.size(-2);
      i64 r2 = _x2.size(-2);
      auto dim1 = _x1.dim();
      auto dim2 = _x2.dim();
      IntArrayRef batch_tensor1(_x1.sizes().data(), dim1 - 2);
      IntArrayRef batch_tensor2(_x2.sizes().data(), dim2 - 2);
      vector<i64> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);
      vector<i64> tensor1_expand_size(expand_batch_portion);
      tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
      vector<i64> tensor2_expand_size(expand_batch_portion);
      tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});

      // Compute the linearized batch size
      const i64 batch_product = multiply_integers(expand_batch_portion);

      // Gracefully handle empty Tensors
      if (r1 == 0 || r2 == 0 || c1 == 0 || batch_product == 0) {
        return zeros_like(_x1, _x1.options());
      }

      Tensor x1 = _x1;
      if (tensor1_expand_size != x1.sizes()) {
        x1 = x1.expand(tensor1_expand_size).contiguous();
      }
      Tensor x2 = _x2;
      if (tensor2_expand_size != x2.sizes()) {
        x2 = x2.expand(tensor2_expand_size).contiguous();
      }

      TORCH_CHECK(x1.is_contiguous(), "_cdist_backward requires X1 to be contiguous");
      TORCH_CHECK(x2.is_contiguous(), "_cdist_backward requires X2 to be contiguous");
      TORCH_CHECK(cdist.is_contiguous(), "_cdist_backward requires dist to be contiguous");
      TORCH_CHECK(grad.is_contiguous(), "_cdist_backward requires grad to be contiguous");
      i64 n = x1.size(-2);
      i64 m = x1.size(-1);
      auto device1 = x1.device().type();
      TORCH_CHECK(device1 == kCPU || device1 == kCUDA, "_cdist_backward only supports CPU and CUDA devices, X1 got: ", device1);
      auto device2 = x2.device().type();
      TORCH_CHECK(device2 == kCPU || device2 == kCUDA, "_cdist_backward only supports CPU and CUDA devices, X2 got: ", device2);

      Tensor grad_x1 =
          empty({batch_product, n, m}, x1.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      cdist_backward_stub(device1, grad_x1, grad, x1, x2, p, cdist);

      // Use x1.size() here and not the original size of _x1.size() as this gradient is not taking broadcasting into account
      // Broadcasting will be handled automatically by the autograd engine
      return grad_x1.view(x1.sizes());
        */
}

pub fn pdist_forward(
        self_: &Tensor,
        p:     f64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.is_contiguous(), "_pdist_forward requires contiguous input");
      auto device = self.device().type();
      TORCH_CHECK(device == kCPU || device == kCUDA, "_pdist_forward only supports CPU and CUDA devices, got: ", device);
      Tensor result = empty({0}, self.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      if (self.size(0) <= 1) {
        result.resize_({0});
      } else {
        i64 n = self.size(0);
        i64 c = n * (n - 1) / 2;
        result.resize_({c});
        if (self.size(1) == 0) {
          result.fill_(0);
        } else {
          pdist_forward_stub(device, result, self, p);
        }
      }
      return result;
        */
}

pub fn pdist_backward(
        grad:  &Tensor,
        self_: &Tensor,
        p:     f64,
        pdist: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.is_contiguous(), "_pdist_backward requires self to be contiguous");
      TORCH_CHECK(pdist.is_contiguous(), "_pdist_backward requires pdist to be contiguous");
      auto device = self.device().type();
      TORCH_CHECK(device == kCPU || device == kCUDA, "_pdist_backward only supports CPU and CUDA devices, got: ", device);
      Tensor result = empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      pdist_backward_stub(device, result, grad, self, p, pdist);
      return result;
        */
}

pub fn cosine_similarity(
        x1:  &Tensor,
        x2:  &Tensor,
        dim: i64,
        eps: f64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(x1.sizes() == x2.sizes(), "cosine_similarity requires both inputs to have the same sizes, but x1 has ",
                  x1.sizes(), " and x2 has ", x2.sizes())
      // Follow scipy impl to improve numerical precision
      // Use x / sqrt(x * x) instead of x / (sqrt(x) * sqrt(x))
      Tensor w12 = sum(x1 * x2, dim);
      Tensor w1 = sum(x1 * x1, dim);
      Tensor w2 = sum(x2 * x2, dim);
      Tensor n12 = (w1 * w2).clamp_min_(eps * eps).sqrt_();
      return w12.div_(n12);
        */
}
