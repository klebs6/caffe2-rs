crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Sorting.h]

#[repr(u8)]
pub enum QUANTILE_INTERPOLATION_MODE {
    LINEAR,
    LOWER,
    HIGHER,
    MIDPOINT,
    NEAREST
}

pub type SortFn = fn(
        values:     &mut Tensor,
        indices:    &mut Tensor,
        dim:        i64,
        descending: bool,
        stable:     bool
) -> ();

pub type TopkFn = fn(
        _0: &Tensor,
        _1: &Tensor,
        _2: &Tensor,
        _3: i64,
        _4: i64,
        _5: bool,
        _6: bool
) -> ();


declare_dispatch!{sort_fn, sort_stub}
declare_dispatch!{topk_fn, topk_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Sorting.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(topk) (
        const Tensor& self,
        i64 k,
        i64 dim_,
        bool largest,
        bool sorted) {

        i64 dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
        TORCH_CHECK(
            k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
            "selected index k out of range");
        i64 sliceSize = self.dim() == 0 ? 1 : self.size(dim);
        TORCH_CHECK(k >= 0 && k <= sliceSize, "k not in range for dimension");

        // Build the output size, which is the dim being selected set to
        // size k
        DimVector topKSize(self.sizes().vec());
        if (topKSize.size() > 0) {
          topKSize[dim] = k;
        }
        set_output(0, topKSize, self.options());
        set_output(1, topKSize, self.options().dtype(kLong));
      }
    */
}

define_dispatch!{sort_stub}
define_dispatch!{topk_stub}

/**
  | Note from TH:
  | 
  | I cut and pasted (slightly adapted)
  | the quicksort code from
  | 
  | Sedgewick's 1978 "Implementing Quicksort
  | Programs" article http://www.csie.ntu.edu.tw/~b93076/p847-sedgewick.pdf
  | 
  | It is the state of the art existing implementation.
  | The macros are here to make as close a
  | match as possible to the pseudocode
  | of
  | 
  | Program 2 p.851
  | 
  | -----------
  | @note
  | 
  | other partition schemes exist, and
  | are typically presented in textbook,
  | but those are less efficient. See e.g.
  | http://cs.stackexchange.com/questions/11458/quicksort-partitioning-hoare-vs-lomuto
  | 
  | Julien, November 12th 2013
  |
  */
pub fn quick_select_template<Scalar, Comp, Fn>(
    arr:       TensorAccessor<Scalar,1>,
    k:         i64,
    gt_or_nan: Comp,
    swap_fn:   Fn)  {

    todo!();
        /*
      i64 P, L, R, i, j;
      Scalar piv;
      L = 0;
      R = arr.size(0) - 1;

      do {
        if (R <= L) // One element only
          return;

        if (R == L + 1) { // Two elements only
          if (gt_or_nan(arr[L], arr[R])) {
            swap_fn(L, R);
          }
          return;
        }

        // Use median of three for pivot choice
        P = (L + R) >> 1;
        swap_fn(P, L + 1);
        if (gt_or_nan(arr[L + 1], arr[R])) {
          swap_fn(L + 1, R);
        }
        if (gt_or_nan(arr[L], arr[R])) {
          swap_fn(L, R);
        }
        if (gt_or_nan(arr[L + 1], arr[L])) {
          swap_fn(L + 1, L);
        }

        i = L + 1;
        j = R;
        piv = arr[L];
        do {
          do
            i++;
          while (gt_or_nan(piv, arr[i]));
          do
            j--;
          while (gt_or_nan(arr[j], piv));
          if (j < i)
            break;
          swap_fn(i, j);
        } while (true);
        swap_fn(L, j);

        // Re-set active partition
        if (j <= k)
          L = i;
        if (j >= k)
          R = j - 1;
      } while (true);
        */
}

pub fn get_quantile_interpolation_mode(interpolation: StringView) -> QuantileInterpolationMode {
    
    todo!();
        /*
            if (interpolation == "linear") {
        return QUANTILE_INTERPOLATION_MODE::LINEAR;
      } else if (interpolation == "lower") {
        return QUANTILE_INTERPOLATION_MODE::LOWER;
      } else if (interpolation == "higher") {
        return QUANTILE_INTERPOLATION_MODE::HIGHER;
      } else if (interpolation == "midpoint") {
        return QUANTILE_INTERPOLATION_MODE::MIDPOINT;
      } else if (interpolation == "nearest") {
        return QUANTILE_INTERPOLATION_MODE::NEAREST;
      } else {
        TORCH_CHECK(
            false,
            "quantile() interpolation must be one of linear, lower, higher, midpoint or nearest, but got ",
            interpolation);
      }
        */
}

pub fn quantile_impl(
        out:           &mut Tensor,
        self_:         &Tensor,
        q:             &Tensor,
        dim:           Option<i64>,
        keepdim:       bool,
        interpolation: &QuantileInterpolationMode,
        ignore_nan:    bool)  {
    
    todo!();
        /*
            i64 dim = maybe_wrap_dim(_dim.value_or(0), self.dim());

      TORCH_CHECK(self.numel() > 0, "quantile() input tensor must be non-empty");
      TORCH_CHECK(q.dim() <= 1, "quantile() q must be a scalar or 1D tensor");
      TORCH_CHECK(
          self.scalar_type() == kFloat || self.scalar_type() == kDouble,
          "quantile() input tensor must be either float or double dtype");
      TORCH_CHECK(
          self.scalar_type() == q.scalar_type(),
          "quantile() q tensor must be same dtype as the input tensor");
      TORCH_CHECK(
          self.scalar_type() == out.scalar_type(),
          "quantile() out tensor must be same dtype as the input tensor");
      TORCH_CHECK(
          self.device() == q.device(),
          "quantile() q tensor must be on the same device as the input tensor");
      TORCH_CHECK(
          self.device() == out.device(),
          "quantile() out tensor must be on the same device as the input tensor");

      // Compute output shape: q_size + reduced_size
      vector<i64> out_shape;
      if (_dim && self.dim() > 0) {
        out_shape = self.sizes().vec();
        if (keepdim) {
          out_shape[dim] = 1;
        } else {
          out_shape.erase(out_shape.begin() + dim);
        }
      } else if (keepdim) {
        out_shape = vector<i64>(self.dim(), 1);
      }
      if (q.dim() > 0) {
        out_shape.insert(out_shape.begin(), q.numel());
      }
      resize_output(out, out_shape);

      // Checks that all q values are between 0 and 1, inclusive
      // NOTE: this check is only performed when running on the CPU to avoid
      // synchronizing an accelerator with the CPU
      if (self.device().is_cpu()) {
        TORCH_CHECK(
            q.ge(0).logical_and_(q.le(1)).all().item<bool>(),
            "quantile() q values must be in the range [0, 1]");
      }

      // Flatten input if no dim provided else move dim to reduce as last dimension.
      // Sort to efficiently query kth values.
      Tensor sorted;
      if (!_dim) {
        sorted = get<0>(self.flatten().sort());
      } else if (dim == self.dim() - 1) {
        sorted = get<0>(self.sort());
      } else {
        sorted = get<0>(self.unsqueeze(-1).transpose(dim, -1).sort());
      }

      // Treat q as a 1D tensor for the following computations
      if (q.dim() == 0) {
        out_shape.insert(out_shape.begin(), q.numel());
      }

      // View input as reduced_size + size of dim to reduce
      vector<i64> in_shape(out_shape.size());
      copy(out_shape.begin() + 1, out_shape.end(), in_shape.begin());
      in_shape[in_shape.size() - 1] = sorted.size(-1);
      sorted = sorted.view(in_shape);

      // Ensure converting from i64 to double won't overflow
      TORCH_CHECK(
          sorted.size(-1) <= pow(2, 24),
          "quantile() input tensor is too large");

      // Convert q in [0, 1] to ranks in [0, reduction_size)
      Tensor ranks;
      if (ignore_nan) {
        // For nanquantile, compute ranks based on number of non-nan values.
        // If all values are nan, set rank to 0 so the quantile computed is nan.
        ranks = q * (sorted.isnan().logical_not_().sum(-1, true) - 1);
        ranks.masked_fill_(ranks < 0, 0);
      } else {
        // For quantile, compute ranks based on reduction size. If there is nan
        // set rank to last index so the quantile computed will be nan.
        i64 last_index = sorted.size(-1) - 1;
        vector<Tensor> tl =
            broadcast_tensors({q * last_index, sorted.isnan().any(-1, true)});
        ranks = masked_fill(tl[0], tl[1], last_index);
      }

      // adjust ranks based on the interpolation mode
      if (interpolation == QUANTILE_INTERPOLATION_MODE::LOWER) {
        ranks.floor_();
      } else if (interpolation == QUANTILE_INTERPOLATION_MODE::HIGHER) {
        ranks.ceil_();
      } else if (interpolation == QUANTILE_INTERPOLATION_MODE::NEAREST) {
        ranks.round_();
      }

      Tensor ranks_below = ranks.toType(kLong);
      Tensor values_below = sorted.gather(-1, ranks_below);

      // Actual interpolation is only needed for the liner and midpoint modes
      if (interpolation == QUANTILE_INTERPOLATION_MODE::LINEAR ||
          interpolation == QUANTILE_INTERPOLATION_MODE::MIDPOINT) {
        // calculate weights for linear and midpoint
        Tensor weights = interpolation == QUANTILE_INTERPOLATION_MODE::MIDPOINT
            ? full_like(ranks, 0.5)
            : ranks - ranks_below;

        // Interpolate to compute quantiles and store in values_below
        Tensor ranks_above = ranks.ceil_().toType(kLong);
        Tensor values_above = sorted.gather(-1, ranks_above);
        values_below.lerp_(values_above, weights);
      }

      if (q.dim() == 0) {
        // If q is scalar, remove last dim to match out shape
        values_below.squeeze_(-1);
      } else {
        // Move quantiles to first dim to match out shape
        values_below.unsqueeze_(0).transpose_(0, -1).squeeze_(-1);
      }

      out.copy_(values_below);
        */
}


pub fn kthvalue_out_impl_cpu(
        values:  &mut Tensor,
        indices: &mut Tensor,
        self_:   &Tensor,
        k:       i64,
        dim:     i64,
        keepdim: bool) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            i64 dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
      zero_numel_check_dims(self, dim, "kthvalue()");

      assert_no_overlap(self, values);

      _reduction_with_indices_allocate_or_resize_output(
          values, indices, self, dim_, keepdim);
      if (self.dim() == 0 && self.numel() == 1) {
        values.copy_(self);
        indices.zero_();
        return forward_as_tuple(values, indices);
      }
      auto tmp_values = self.clone(MemoryFormat::Contiguous);
      auto tmp_indices = empty(self.sizes(), self.options().dtype(kLong));

      auto tmp_values_stride = tmp_values.strides()[dim];
      auto tmp_indices_stride = tmp_indices.strides()[dim];
      auto sizes = self.sizes();

      TORCH_CHECK(indices.scalar_type() == kLong);

      auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .declare_static_shape(sizes, /*squash_dims=*/dim)
        .add_output(tmp_values)
        .add_output(tmp_indices)
        .add_output(values)
        .add_output(indices)
        .build();

      AT_DISPATCH_ALL_TYPES(self.scalar_type(), "kthvalue_cpu", [&] {
        auto loop = [&](char** data, const i64* strides, i64 n) {
          for (i64 i = 0; i < n; ++i) {
            TensorAccessor<Scalar, 1> tmp_values(
                reinterpret_cast<Scalar*>(data[0] + i * strides[0]),
                &sizes[dim], &tmp_values_stride);
            TensorAccessor<i64, 1> tmp_indices(
                reinterpret_cast<i64*>(data[1] + i * strides[1]),
                &sizes[dim], &tmp_indices_stride);
            auto mode_value = reinterpret_cast<Scalar*>(data[2] + i * strides[2]);
            auto mode_index = reinterpret_cast<i64*>(data[3] + i * strides[3]);

            for (i64 j = 0; j < tmp_indices.size(0); j++) {
              tmp_indices[j] = j;
            }

            // we want NaN to be sorted as top for numpy compatibility
            quick_select_template(
              tmp_values,
              k - 1,
              [](Scalar x, Scalar y) -> bool {
                return (
                  (_isnan<Scalar>(x) && !_isnan<Scalar>(y)) || (x > y));
              },
              [&](i64 i, i64 j) {
                swap(tmp_values[i], tmp_values[j]);
                swap(tmp_indices[i], tmp_indices[j]);
              });
            *mode_value = tmp_values[k - 1];
            *mode_index = tmp_indices[k - 1];
          }
        };

        i64 grain_size = internal::GRAIN_SIZE / max(i64{1}, sizes[dim]);
        iter.for_each(loop, /*grain_size=*/grain_size);
      });

      if (!keepdim) {
        values.squeeze_(dim);
        indices.squeeze_(dim);
      }
      return forward_as_tuple(values, indices);
        */
}

/**
  | Computes both the median and its index
  | along dimension dim of the input
  |
  */
pub fn median_with_indices_impl(
        values:     &mut Tensor,
        indices:    &mut Tensor,
        self_:      &Tensor,
        dim:        i64,
        keepdim:    bool,
        ignore_nan: bool) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            dim = maybe_wrap_dim(dim, self.dim());

      i64 size = self.dim() > 0 ? self.size(dim) : 1;
      zero_numel_check_dims(self, dim, "median()");

      checkDeviceType("median", {values, indices}, self.device().type());
      checkScalarType("median", {indices, "indices", 1}, kLong);
      checkSameType("median", {values, "values", 0}, {self, "self", 2});

      vector<i64> out_shape = self.sizes().vec();
      if (self.dim() > 0) {
        if (keepdim) {
          out_shape[dim] = 1;
        } else {
          out_shape.erase(out_shape.begin() + dim);
        }
      }

      resize_output(values, out_shape);
      resize_output(indices, out_shape);

      // Ensure #dim is the same for all tensors required for dim_apply
      Tensor in = self.dim() > 0 ? self : self.unsqueeze(0);
      Tensor vals = keepdim && self.dim() > 0 ? values : values.unsqueeze(dim);
      Tensor inds = keepdim && self.dim() > 0 ? indices : indices.unsqueeze(dim);

      // Make dim to reduce contiguous (stride=1)
      if (in.stride(dim) > 1) {
        in = in.unsqueeze(-1).transpose_(dim, -1).squeeze_(dim).contiguous();
        vals = vals.unsqueeze(-1).transpose_(dim, -1).squeeze_(dim);
        inds = inds.unsqueeze(-1).transpose_(dim, -1).squeeze_(dim);
        dim = in.dim() - 1;
      }

      auto sizes = in.sizes();
      auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .declare_static_shape(sizes, /*squash_dims=*/dim)
        .add_output(vals)
        .add_output(inds)
        .add_input(in)
        .build();

      AT_DISPATCH_ALL_TYPES(in.scalar_type(), "median_out", [&] {
        auto loop = [&](char** data, const i64* strides, i64 n) {
          for (i64 i = 0; i < n; ++i) {
            auto valp = reinterpret_cast<Scalar*>(data[0] + i * strides[0]);
            auto indp = reinterpret_cast<i64*>(data[1] + i * strides[1]);
            auto ip = reinterpret_cast<const Scalar*>(data[2] + i * strides[2]);

            // For torch.median, search for NaN and return it if found
            if (!ignore_nan) {
              const Scalar* nanp = find_if(ip, ip + size, _isnan<Scalar>);
              if (nanp != ip + size) {
                *valp = *nanp;
                *indp = nanp - ip;
                continue;
              }
            }

            // Vector of indices for indirectly partitioning input around median
            vector<i64> idx(size);
            auto first = idx.begin();
            auto last = idx.end();
            iota(first, last, 0);

            // We partition the input around the median indirectly using the indices
            // vector so that nth points to the index of the median in the unmodified
            // input tensor.
            auto nth = first;
            if (!ignore_nan) {
              // If we got here, there are no nan values
              nth += (size - 1) / 2;
              nth_element(first, nth, last, [&ip](i64 i, i64 j) {
                return ip[i] < ip[j] || (ip[i] == ip[j] && i < j);
              });
            } else {
              // For torch.nanmedian, compute median of non-nan values only
              i64 num_nan = count_if(ip, ip + size, _isnan<Scalar>);
              nth += (size - num_nan - 1) / 2;
              nth_element(first, nth, last, [&ip](i64 i, i64 j) {
                return ip[i] < ip[j] || (ip[i] == ip[j] && i < j) ||
                    (_isnan(ip[j]) && !_isnan(ip[i]));
              });
            }

            *valp = ip[*nth];
            *indp = *nth;
          }
        };
        i64 grain_size = internal::GRAIN_SIZE / max(i64{1}, sizes[dim]);
        iter.for_each(loop, /*grain_size=*/grain_size);
      });

      return forward_as_tuple(values, indices);
        */
}

/**
  | Computes the median of all values in
  | the input
  |
  */
pub fn median_impl(
        self_:      &Tensor,
        ignore_nan: bool) -> Tensor {
    
    todo!();
        /*
            NoNamesGuard guard;

      // Clone the input tensor so we can partition it around the median value
      Tensor in = self.clone();
      Tensor out = empty({}, self.options());
      const i64 size = self.numel();

      AT_DISPATCH_ALL_TYPES(in.scalar_type(), "median_cpu", [&] {
        Scalar* op = out.data_ptr<Scalar>();
        Scalar* first = in.data_ptr<Scalar>();
        Scalar* last = first + size;

        // For torch.median, if there are nan values return nan
        if (!ignore_nan && any_of(first, last, _isnan<Scalar>)) {
          *op = numeric_limits<Scalar>::quiet_NaN();
          return;
        }

        Scalar* median = first;
        if (!ignore_nan) {
          // If we got here, there are no nan values
          median += (size - 1) / 2;
          nth_element(first, median, last);
        } else {
          // For torch.nanmedian, compute median of non-nan values only
          i64 num_nan = count_if(first, last, _isnan<Scalar>);
          median += (size - num_nan - 1) / 2;
          nth_element(first, median, last, [](Scalar a, Scalar b) {
            return a < b || (_isnan(b) && !_isnan(a));
          });
        }

        *op = *median;
      });

      return out;
        */
}

pub fn quantile_out_a(
        self_:         &Tensor,
        q:             &Tensor,
        dim:           Option<i64>,
        keepdim:       bool,
        interpolation: StringView,
        out:           &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            quantile_impl(
          out,
          self,
          q,
          move(dim),
          keepdim,
          get_quantile_interpolation_mode(interpolation),
          /*ignore_nan=*/false);
      return out;
        */
}


pub fn quantile_out_b(
        self_:         &Tensor,
        q:             f64,
        dim:           Option<i64>,
        keepdim:       bool,
        interpolation: StringView,
        out:           &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);
      return native::quantile_out(
          self,
          scalar_tensor(q, self.options()),
          move(dim),
          keepdim,
          interpolation,
          out);
        */
}


pub fn quantile_a(
        self_:         &Tensor,
        q:             &Tensor,
        dim:           Option<i64>,
        keepdim:       bool,
        interpolation: StringView) -> Tensor {
    
    todo!();
        /*
            Tensor out = empty({0}, self.options());
      quantile_impl(
          out,
          self,
          q,
          move(dim),
          keepdim,
          get_quantile_interpolation_mode(interpolation),
          /*ignore_nan=*/false);
      return out;
        */
}


pub fn quantile_b(
        self_:         &Tensor,
        q:             f64,
        dim:           Option<i64>,
        keepdim:       bool,
        interpolation: StringView) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);
      return native::quantile(
          self, scalar_tensor(q, self.options()), move(dim), keepdim, interpolation);
        */
}


pub fn nanquantile_out_a(
        self_:         &Tensor,
        q:             &Tensor,
        dim:           Option<i64>,
        keepdim:       bool,
        interpolation: StringView,
        out:           &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            quantile_impl(
          out,
          self,
          q,
          move(dim),
          keepdim,
          get_quantile_interpolation_mode(interpolation),
          /*ignore_nan=*/true);
      return out;
        */
}


pub fn nanquantile_out_b(
        self_:         &Tensor,
        q:             f64,
        dim:           Option<i64>,
        keepdim:       bool,
        interpolation: StringView,
        out:           &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);
      return native::nanquantile_out(
          self,
          scalar_tensor(q, self.options()),
          move(dim),
          keepdim,
          interpolation,
          out);
        */
}


pub fn nanquantile_a(
        self_:         &Tensor,
        q:             &Tensor,
        dim:           Option<i64>,
        keepdim:       bool,
        interpolation: StringView) -> Tensor {
    
    todo!();
        /*
            Tensor out = empty({0}, self.options());
      quantile_impl(
          out,
          self,
          q,
          move(dim),
          keepdim,
          get_quantile_interpolation_mode(interpolation),
          /*ignore_nan=*/true);
      return out;
        */
}


pub fn nanquantile_b(
        self_:         &Tensor,
        q:             f64,
        dim:           Option<i64>,
        keepdim:       bool,
        interpolation: StringView) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);
      return native::nanquantile(
          self, scalar_tensor(q, self.options()), move(dim), keepdim, interpolation);
        */
}


pub fn quantile_out_c(
        self_:   &Tensor,
        q:       &Tensor,
        dim:     Option<i64>,
        keepdim: bool,
        out:     &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
      return native::quantile_out(self, q, move(dim), keepdim, "linear", out);
        */
}


pub fn quantile_out_d(
        self_:   &Tensor,
        q:       f64,
        dim:     Option<i64>,
        keepdim: bool,
        out:     &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
      return native::quantile_out(self, q, move(dim), keepdim, "linear", out);
        */
}


pub fn quantile_c(
        self_:   &Tensor,
        q:       &Tensor,
        dim:     Option<i64>,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
      return native::quantile(self, q, move(dim), keepdim, "linear");
        */
}


pub fn quantile_d(
        self_:   &Tensor,
        q:       f64,
        dim:     Option<i64>,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
      return native::quantile(self, q, move(dim), keepdim, "linear");
        */
}


pub fn nanquantile_out_c(
        self_:   &Tensor,
        q:       &Tensor,
        dim:     Option<i64>,
        keepdim: bool,
        out:     &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
      return native::nanquantile_out(self, q, move(dim), keepdim, "linear", out);
        */
}


pub fn nanquantile_out_d(
        self_:   &Tensor,
        q:       f64,
        dim:     Option<i64>,
        keepdim: bool,
        out:     &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
      return native::nanquantile_out(self, q, move(dim), keepdim, "linear", out);
        */
}


pub fn nanquantile_c(
        self_:   &Tensor,
        q:       &Tensor,
        dim:     Option<i64>,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
      return native::nanquantile(self, q, move(dim), keepdim, "linear");
        */
}


pub fn nanquantile_d(
        self_:   &Tensor,
        q:       f64,
        dim:     Option<i64>,
        keepdim: bool) -> Tensor {
    
    todo!();
        /*
      return native::nanquantile(self, q, move(dim), keepdim, "linear");
        */
}


pub fn kthvalue_out_cpu(
        self_:   &Tensor,
        k:       i64,
        dim:     i64,
        keepdim: bool,
        values:  &mut Tensor,
        indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        return kthvalue_out_impl_cpu(values, indices, self, k, dim, keepdim);
      }();
      namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
      namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
      return result;
        */
}


pub fn kthvalue_out(
        self_:   &Tensor,
        k:       i64,
        dim:     Dimname,
        keepdim: bool,
        values:  &mut Tensor,
        indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            return kthvalue_out(
          values, indices, self, k, dimname_to_position(self, dim), keepdim);
        */
}


pub fn kthvalue_a(
    self_:   &Tensor,
    k:       i64,
    dim:     i64,
    keepdim: bool) -> (Tensor,Tensor) {

    todo!();
        /*
            Tensor values = empty({0}, self.options());
      Tensor indices = empty({0}, self.options().dtype(kLong));
      kthvalue_out(values, indices, self, k, dim, keepdim);
      return make_tuple(values, indices);
        */
}

pub fn kthvalue_b(
    self_:   &Tensor,
    k:       i64,
    dim:     Dimname,
    keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return kthvalue(self, k, dimname_to_position(self, dim), keepdim);
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(topk_out_cpu)
       (const Tensor& self,
        i64 k,
        i64 dim_,
        bool largest,
        bool sorted,
        const Tensor& values,
        const Tensor& indices) {
      i64 dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
      TORCH_CHECK(
          k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
          "selected index k out of range");

      if (self.dim() == 0 && self.numel() == 1) {
        values.copy_(self);
        indices.zero_();
      } else {
        topk_stub(kCPU, values, indices, self, k, dim, largest, sorted);
      }
    }
    */
}


pub fn median_out_cpu(
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool,
        values:  &mut Tensor,
        indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        return median_with_indices_impl(
            values, indices, self, dim, keepdim, /*ignore_nan=*/false);
      }();
      namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
      namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
      return result;
        */
}


pub fn median_out(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool,
        values:  &mut Tensor,
        indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            return median_out(
          values, indices, self, dimname_to_position(self, dim), keepdim);
        */
}


pub fn median_a(
    self_:   &Tensor,
    dim:     i64,
    keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            Tensor values = empty({0}, self.options());
      Tensor indices = empty({0}, self.options().dtype(kLong));
      median_out(values, indices, self, dim, keepdim);
      return make_tuple(values, indices);
        */
}

pub fn median_b(
    self_:   &Tensor,
    dim:     Dimname,
    keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return median(self, dimname_to_position(self, dim), keepdim);
        */
}


pub fn median_cpu(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return median_impl(self, /*ignore_nan=*/false);
        */
}


pub fn nanmedian_out_cpu(
        self_:   &Tensor,
        dim:     i64,
        keepdim: bool,
        values:  &mut Tensor,
        indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        return median_with_indices_impl(
            values, indices, self, dim, keepdim, /*ignore_nan=*/true);
      }();
      namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
      namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
      return result;
        */
}


pub fn nanmedian_out(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool,
        values:  &mut Tensor,
        indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            return nanmedian_out(
          values, indices, self, dimname_to_position(self, dim), keepdim);
        */
}

pub fn nanmedian_a(
    self_:   &Tensor,
    dim:     i64,
    keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            Tensor values = empty({0}, self.options());
      Tensor indices = empty({0}, self.options().dtype(kLong));
      nanmedian_out(values, indices, self, dim, keepdim);
      return make_tuple(values, indices);
        */
}

pub fn nanmedian_b(
    self_:   &Tensor,
    dim:     Dimname,
    keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return nanmedian(self, dimname_to_position(self, dim), keepdim);
        */
}

pub fn nanmedian_cpu(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return median_impl(self, /*ignore_nan=*/true);
        */
}


pub fn sort_out_cpu_stable(
        self_:      &Tensor,
        stable:     Option<bool>,
        dim:        i64,
        descending: bool,
        values:     &mut Tensor,
        indices:    &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            values.resize_(self.sizes()).copy_(self);
      indices.resize_(self.sizes());

      // check if self is scalar
      if (self.dim() == 0 && self.numel() == 1) {
        indices.zero_();
        return forward_as_tuple(values, indices);
      }

      TORCH_INTERNAL_ASSERT(stable.has_value(), "sort_out(): optional<bool> for stable has to have value.");
      sort_stub(kCPU, values, indices, dim, descending, stable.value());

      return forward_as_tuple(values, indices);
        */
}


pub fn sort_out_cpu(
        self_:      &Tensor,
        dim:        i64,
        descending: bool,
        values:     &mut Tensor,
        indices:    &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            return native::sort_out_cpu_stable(
          self, /*stable=*/false, dim, descending, values, indices);
        */
}


pub fn sort_cpu_stable(
        self_:      &Tensor,
        stable:     Option<bool>,
        dim:        i64,
        descending: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_complex(), "sort(): input tensor must be of non-complex type");
      Tensor values = empty({0}, self.options());
      Tensor indices = empty({0}, self.options().dtype(kLong));
      return native::sort_out_cpu_stable(self, stable, dim, descending, values, indices);
        */
}


pub fn sort_cpu(
        self_:      &Tensor,
        dim:        i64,
        descending: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return sort_cpu_stable(self, /*stable=*/false, dim, descending);
        */
}


pub fn msort_out(
        self_:  &Tensor,
        values: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            Tensor indices = empty({0}, self.options().dtype(kLong));
      sort_out(values, indices, self, 0, false);
      return values;
        */
}


pub fn msort(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return get<0>(sort(self, 0, false));
        */
}


pub fn argsort(
        self_:      &Tensor,
        dim:        i64,
        descending: bool) -> Tensor {
    
    todo!();
        /*
            return get<1>(sort(self, dim, descending));
        */
}
