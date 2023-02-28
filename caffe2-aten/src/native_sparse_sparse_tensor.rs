/*!
  | Basic functions on sparse tensors
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/sparse/SparseTensor.cpp]

/******************************************************************************
 * access methods
 ******************************************************************************/

pub trait SparseDimSparse {
    fn sparse_dim_sparse(&self) -> i64;
}

pub trait DenseDimSparse {

    fn dense_dim_sparse(&self) -> i64;
}

pub trait IsCoalescedSparse {

    fn is_coalesced_sparse(&self) -> bool;
}

pub trait NnzSparse {

    fn nnz_sparse(&self) -> i64;
}

pub trait ValuesSparse {
    fn values_sparse(&self) -> Tensor;
}

pub trait CoalescedSparse {

    fn coalesced_sparse(
        &mut self,
        coalesced: bool) -> &mut Tensor;
}

pub trait IndicesSparse {
    fn indices_sparse(&self) -> Tensor;
}

pub trait Coalesce {

    fn coalesce(&self) -> SparseTensor;
}

pub trait CoalesceSparseCpu {

    fn coalesce_sparse_cpu(&self) -> SparseTensor;
}

impl SparseDimSparse for SparseTensor {

    fn sparse_dim_sparse(&self) -> i64 {
        
        todo!();
            /*
                return get_sparse_impl(self)->sparse_dim();
            */
    }
}

impl DenseDimSparse for SparseTensor {

    fn dense_dim_sparse(&self) -> i64 {
        
        todo!();
            /*
                return get_sparse_impl(self)->dense_dim();
            */
    }
}

impl IsCoalescedSparse for SparseTensor {

    fn is_coalesced_sparse(&self) -> bool {
        
        todo!();
            /*
                return get_sparse_impl(self)->coalesced();
            */
    }
}

impl NnzSparse for SparseTensor {

    fn nnz_sparse(&self) -> i64 {
        
        todo!();
            /*
                return get_sparse_impl(self)->nnz();
            */
    }
}

impl IndicesSparse for SparseTensor {

    /**
      | Why are there so many methods to get indices
      | and value?
      |
      | See Note [ Sparse: different methods to get
      | indices and values ] in native_functions.yaml
      */
    fn indices_sparse(&self) -> Tensor {
        
        todo!();
            /*
                return get_sparse_impl(self)->indices();
            */
    }
}

impl ValuesSparse for SparseTensor {

    fn values_sparse(&self) -> Tensor {
        
        todo!();
            /*
                return get_sparse_impl(self)->values();
            */
    }
}

impl CoalescedSparse for SparseTensor {

    fn coalesced_sparse(
        &mut self,
        coalesced: bool) -> &mut Tensor {
        
        todo!();
            /*
                get_sparse_impl(self)->set_coalesced(coalesced);
          return self;
            */
    }
}

impl IndicesSparse for Tensor {

    fn indices_sparse(&self) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(
              self.is_coalesced(),
              "Cannot get indices on an uncoalesced tensor, please call .coalesce() first");
          return get_sparse_impl(self)->indices().alias();
            */
    }
}

impl ValuesSparse for Tensor {

    fn values_sparse(&self) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(
              self.is_coalesced(),
              "Cannot get values on an uncoalesced tensor, please call .coalesce() first");
          return get_sparse_impl(self)->values().alias();
            */
    }
}

/**
  | creation methods
  | 
  | See NOTE [ Sparse: autograd and API ]
  | for details
  |
  */
pub fn new_sparse(
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> SparseTensor {
    
    todo!();
        /*
            AT_ASSERT(layout.has_value() && *layout == kSparse);
      DispatchKey dispatch_key;
      if (device_or_default(device).is_cuda()) {
        dispatch_key = DispatchKey::SparseCUDA;
      } else if (device_or_default(device).is_xpu()) {
        dispatch_key = DispatchKey::SparseXPU;
      } else {
        dispatch_key = DispatchKey::SparseCPU;
      }
      return make_tensor<SparseTensorImpl>(
          DispatchKeySet(dispatch_key),
          scalarTypeToTypeMeta(dtype_or_default(dtype)));
        */
}

/* ----- Actual dispatched creation methods ** ----- */

pub fn new_with_dims_sparse(
        sparse_dim: i64,
        dense_dim:  i64,
        size:       &[i64],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> SparseTensor {
    
    todo!();
        /*
            SparseTensor self = new_sparse(dtype, layout, device, pin_memory);
      get_sparse_impl(self)->resize_and_clear_(sparse_dim, dense_dim, size);
      return self;
        */
}

pub fn new_with_dims_and_tensor_sparse(
    sparse_dim: i64,
    dense_dim:  i64,
    size:       &[i64],
    indices:    &Tensor,
    values:     &Tensor,
    dtype:      Option<ScalarType>,
    layout:     Option<Layout>,
    device:     Option<Device>,
    pin_memory: Option<bool>) -> SparseTensor {
    
    todo!();
        /*
            SparseTensor self = new_sparse(dtype, layout, device, pin_memory);
      get_sparse_impl(self)->resize_(sparse_dim, dense_dim, size);
      // NOTE: There is no guarantee that `indices` and `values` don't contain
      // AutogradMeta. However, we want to maintain the invariant that `indices_`
      // and `values_` of a sparse tensor don't contain AutogradMeta, and to achieve
      // that we shallow-copy `indices` and `values` here.
      auto indices_shallow_copy =
          Tensor(indices.unsafeGetTensorImpl()->shallow_copy_and_detach(
              /*version_counter=*/indices.unsafeGetTensorImpl()->version_counter(),
              /*allow_tensor_metadata_change=*/true));
      auto values_shallow_copy =
          Tensor(values.unsafeGetTensorImpl()->shallow_copy_and_detach(
              /*version_counter=*/values.unsafeGetTensorImpl()->version_counter(),
              /*allow_tensor_metadata_change=*/true));
      alias_into_sparse(self, indices_shallow_copy, values_shallow_copy);
      return self;
        */
}

/** Public creation API that dispatch to methods above **/

/**
  | Empty init *
  |
  */
pub fn empty_sparse(
    size:                   &[i32],
    dtype:                  Option<ScalarType>,
    layout:                 Option<Layout>,
    device:                 Option<Device>,
    pin_memory:             Option<bool>,
    optional_memory_format: Option<MemoryFormat>) -> Tensor {

    todo!();
        /*
            TORCH_CHECK(
          !pin_memory.has_value() || !*pin_memory,
          "Only dense CPU tensors can be pinned");
      return new_with_dims_sparse(
          size.size(), 0, size, dtype, layout, device, pin_memory);
        */
}

/**
  | Shape init
  |
  */
pub fn sparse_coo_tensor_a(
    size:       &[i32],
    dtype:      Option<ScalarType>,
    layout:     Option<Layout>,
    device:     Option<Device>,
    pin_memory: Option<bool>) -> Tensor {

    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      return _sparse_coo_tensor_with_dims(size.size(), 0, size, options.layout(kSparse));
        */
}

/* ---------------- Pointer-copy init  ---------------- */

#[inline] pub fn expand_values_if_needed(values: &Tensor) -> Tensor {
    
    todo!();
        /*
            // expand
      if (values.dim() == 0) {
        // Mimic Numpy behavior here and treat it as a 1D tensor
        return values.expand({1});
      } else {
        return values;
      }
        */
}

pub fn sparse_coo_tensor_b(
        indices:    &Tensor,
        values:     &Tensor,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      Tensor values = expand_values_if_needed(values_);

      // arg checking
      TORCH_CHECK(
          !options.has_layout() || options.layout() == kSparse,
          "expected sparse layout, but got layout ",
          options.layout());
      // the following checks are redundant because they are also checked in
      // SparseTensorImpl::set_indices_and_values_unsafe but we need to ensure them
      // in order to infer the shape.
      TORCH_CHECK(
          indices.dim() == 2,
          "indices must be sparse_dim x nnz, but got: ",
          indices.sizes())
      TORCH_CHECK(
          !indices.is_sparse(),
          "expected indices to be a dense tensor, but got indices of layout ",
          indices.layout());

      // If sizes are not given, it is inferred as max index of each dim.
      i64 sparse_dim = indices.size(0);
      i64 dense_dim = values.dim() - 1;

      vector<i64> computed_sizes(sparse_dim + dense_dim);
      if (indices.numel() > 0) {
        // If the indices has elements in it, we infer the minimum sparse dimension
        // sizes as the max value of each dim in indices. NB: It used to keepdim. I
        // think that was wrong.
        Tensor min_indices =
            get</* values */ 0>(indices.min(/* dim */ 1, /* keepdim */ false));
        Tensor computed_indices_sizes =
            get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
        computed_indices_sizes.add_(1); // len = max_index + 1
        Tensor cpu_min_indices = min_indices.to(DeviceType_CPU);
        Tensor cpu_computed_indices_sizes =
            computed_indices_sizes.to(DeviceType_CPU);
        auto cpu_min_indices_accessor = cpu_min_indices.accessor<i64, 1>();
        auto cpu_computed_indices_sizes_accessor =
            cpu_computed_indices_sizes.accessor<i64, 1>();
        for (i64 d = 0; d < sparse_dim; d++) {
          i64 min_index_in_dim = cpu_min_indices_accessor[d];
          TORCH_CHECK(
              min_index_in_dim >= 0,
              "found negative index ",
              min_index_in_dim,
              " for dim ",
              d);
          computed_sizes[static_cast<usize>(d)] =
              cpu_computed_indices_sizes_accessor[d];
        }
      } else {
        // If the indices doesn't have elements in it, there is not enough
        // information to know what the minimum sparse dimension sizes should be,
        // and in this case we set them to 0
        for (i64 d = 0; d < sparse_dim; d++) {
          computed_sizes[static_cast<usize>(d)] = 0;
        }
      }
      for (i64 d = 0; d < dense_dim; d++) {
        computed_sizes[static_cast<usize>(sparse_dim + d)] = values.size(d + 1);
      }

      return _sparse_coo_tensor_with_dims_and_tensors(
          sparse_dim,
          dense_dim,
          computed_sizes,
          indices,
          values,
          values.options().layout(kSparse));
        */
}

pub fn validate_sparse_coo_tensor_args(
        indices: &Tensor,
        values:  &Tensor,
        size:    &[i64])  {
    
    todo!();
        /*
            Tensor values = expand_values_if_needed(values_);

      // the following checks are redundant because they are also checked in
      // SparseTensorImpl::set_indices_and_values_unsafe but we need to ensure them
      // in order to infer the shape.
      TORCH_CHECK(
          indices.dim() == 2,
          "indices must be sparse_dim x nnz, but got: ",
          indices.sizes())
      TORCH_CHECK(
          !indices.is_sparse(),
          "expected indices to be a dense tensor, but got indices of layout ",
          indices.layout());
      i64 sparse_dim = indices.size(0);
      i64 dense_dim = values.dim() - 1;
      TORCH_CHECK(
          size.size() == sparse_dim + dense_dim,
          "number of dimensions must be sparse_dim (",
          sparse_dim,
          ") + dense_dim (",
          dense_dim,
          "), but got ",
          size.size());

      // Check to make sure all indices are within the boundaries of `size`
      if (indices.numel() > 0) {
        Tensor min_indices =
            get</* values */ 0>(indices.min(/* dim */ 1, /* keepdim */ false));
        Tensor max_indices =
            get</* values */ 0>(indices.max(/* dim */ 1, /* keepdim */ false));
        Tensor cpu_min_indices, cpu_max_indices;
        if (indices.is_cuda()) {
          cpu_min_indices = min_indices.to(DeviceType_CPU);
          cpu_max_indices = max_indices.to(DeviceType_CPU);
        } else {
          cpu_min_indices = min_indices;
          cpu_max_indices = max_indices;
        }
        auto cpu_min_indices_accessor = cpu_min_indices.accessor<i64, 1>();
        auto cpu_max_indices_accessor = cpu_max_indices.accessor<i64, 1>();
        for (i64 d = 0; d < sparse_dim; d++) {
          // NB: This used to sync ndim times to access each entry; now we copy
          // everything to CPU first and then access it.
          i64 min_index_in_dim = cpu_min_indices_accessor[d];
          TORCH_CHECK(
              min_index_in_dim >= 0,
              "found negative index ",
              min_index_in_dim,
              " for dim ",
              d);
          i64 max_index_in_dim = cpu_max_indices_accessor[d];
          i64 dim_size = size[static_cast<usize>(d)];
          TORCH_CHECK(
              max_index_in_dim < dim_size,
              "size is inconsistent with indices: for dim ",
              d,
              ", size is ",
              dim_size,
              " but found index ",
              max_index_in_dim);
        }
      }
        */
}

/**
  | NB: Got rid of the sizes == NULL case
  |
  */
pub fn sparse_coo_tensor_c(
        indices:    &Tensor,
        values:     &Tensor,
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
      // arg checking
      TORCH_CHECK(
          !options.has_layout() || options.layout() == kSparse,
          "expected sparse layout, but got layout ",
          options.layout());

      native::_validate_sparse_coo_tensor_args(indices, values, size);
      return native::_sparse_coo_tensor_unsafe(
          indices,
          values,
          size,
          optTypeMetaToScalarType(options.dtype_opt()),
          options.layout_opt(),
          options.device_opt(),
          options.pinned_memory_opt());
        */
}

/**
  | NOTE: _sparse_coo_tensor_unsafe() differs from
  | sparse_coo_tensor() in that we don't check
  | whether any indices are out of boundaries of
  | `size`, thus avoiding a copy from CUDA to CPU.
  |
  | However, this function should ONLY be used
  | where we know that the indices are guaranteed
  | to be within bounds or if the caller is going
  | to call _validate_sparse_coo_tensor_args before
  | using the tensor.
  |
  | NB: Got rid of the size == NULL case
  */
pub fn sparse_coo_tensor_unsafe(
    indices:    &Tensor,
    values:     &Tensor,
    size:       &[i32],
    dtype:      Option<ScalarType>,
    layout:     Option<Layout>,
    device:     Option<Device>,
    pin_memory: Option<bool>) -> Tensor {

    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]

      Tensor values = expand_values_if_needed(values_);

      i64 sparse_dim = indices.size(0);
      i64 dense_dim = values.dim() - 1;

      return _sparse_coo_tensor_with_dims_and_tensors(
          sparse_dim,
          dense_dim,
          size,
          indices,
          values,
          values.options().layout(kSparse));
        */
}

/**
  | NB: Deleted newWithSizeNd variants
  |
  */
pub fn clone_sparse(
    self_:                  &SparseTensor,
    optional_memory_format: Option<MemoryFormat>) -> SparseTensor {

    todo!();
        /*
            TORCH_CHECK(
          !optional_memory_format.has_value(),
          "unsupported memory format option ",
          optional_memory_format.value());
      SparseTensor other = new_with_dims_sparse(
          self.sparse_dim(),
          self.dense_dim(),
          self.sizes(),
          optTypeMetaToScalarType(self.options().dtype_opt()),
          self.options().layout_opt(),
          self.options().device_opt(),
          self.options().pinned_memory_opt());
      copy_into_sparse(other, self._indices(), self._values(), true);
      return other._coalesced_(self.is_coalesced());
        */
}

/******************************************************************************
 * reshaping methods
 ******************************************************************************/

pub fn sparse_resize(
    self_:      &SparseTensor,
    size:       &[i64],
    sparse_dim: i64,
    dense_dim:  i64) -> &SparseTensor {
    
    todo!();
        /*
            get_sparse_impl(self)->resize_(sparse_dim, dense_dim, size);
      return self;
        */
}


pub fn sparse_resize_and_clear(
        self_:      &SparseTensor,
        size:       &[i64],
        sparse_dim: i64,
        dense_dim:  i64) -> &SparseTensor {
    
    todo!();
        /*
            get_sparse_impl(self)->resize_and_clear_(sparse_dim, dense_dim, size);
      return self;
        */
}

pub fn is_same_size_as_sparse(
    self_: &SparseTensor,
    src:   &SparseTensor) -> bool {

    todo!();
        /*
            return self.sparse_dim() == src.sparse_dim() &&
          self.dense_dim() == src.dense_dim() && self.sizes().equals(src.sizes());
        */
}

/**
  | Invoked from native/Resize.cpp (no
  | dynamic dispatch necessary)
  |
  */
pub fn resize_as_sparse(
        self_: &SparseTensor,
        src:   &SparseTensor) -> &SparseTensor {
    
    todo!();
        /*
            if (!_is_same_size_as_sparse(self, src)) {
        sparse_resize_(self, src.sizes(), src.sparse_dim(), src.dense_dim());
      }
      return self;
        */
}

pub fn dense_to_sparse_a(self_: &Tensor) -> SparseTensor {
    
    todo!();
        /*
            return dense_to_sparse(self, self.dim());
        */
}

pub fn dense_to_sparse_b(
    self_:      &Tensor,
    sparse_dim: i64) -> SparseTensor {
    
    todo!();
        /*
            i64 dims = self.dim();
      // TODO: it seems like sparse_dim == 0 could be supported even if self.dim() >
      // 0, but this would take some work and doesn't seem particularly useful.
      TORCH_CHECK(
          sparse_dim > 0 || self.dim() == 0,
          "sparse_dim must be >0 if dimensionality > 0");
      TORCH_CHECK(
          sparse_dim <= dims,
          "sparse_dim must be less than or equal to self.dim()");
      TensorOptions sparse_options = self.options().layout(kSparse);
      vector<i64> sizes = self.sizes().vec();

      Tensor nz = self.nonzero().transpose(0, 1);
      if (nz.size(1) == 0) {
        return new_with_dims_sparse(
            sparse_dim,
            dims - sparse_dim,
            sizes,
            optTypeMetaToScalarType(sparse_options.dtype_opt()),
            sparse_options.layout_opt(),
            sparse_options.device_opt(),
            sparse_options.pinned_memory_opt());
      }
      Tensor indices;
      if (sparse_dim == dims) {
        indices = nz.clone();
      } else {
        Tensor i = nz.narrow(0, 0, sparse_dim);
        tie(indices, ignore, ignore) = unique_dim(i, 1);
        indices = indices.contiguous(); // many sparse CUDA kernels require
                                        // contiguity, see issue #12633
      }

      Tensor values;
      if (self.dim() > 0) {
        auto ix = toListOfOptionalTensors(indices.chunk(indices.size(0), 0));
        values = self.index(ix).squeeze(0).clone(MemoryFormat::Preserve);
      } else {
        AT_ASSERT(nz.sizes().equals({0, 1}));
        // In this cases, indices is a clone of nz, which is a tensor of shape (0,
        // 1). Given sparse tensor invariants, values should be shape (1,)
        values = self.unsqueeze(0).clone(MemoryFormat::Preserve);
      }

      Tensor sparse = sparse_coo_tensor(indices, values, sizes, sparse_options);
      return sparse._coalesced_(true);
        */
}

// NB: Dropped the resizeNd variants
pub fn sparse_to_dense(
    self_: &SparseTensor,
    dtype: Option<ScalarType>) -> Tensor {

    todo!();
        /*
            TORCH_CHECK(
          !dtype.has_value(), "dtype argument is not supported by sparse_to_dense");
      if (self.scalar_type() == ScalarType::Half &&
          self.options().device().is_cpu()) {
        TORCH_CHECK(false, "to_dense() not supported for float16 on CPU");
      }
      Tensor dst = zeros(self.sizes(), self.options().layout(kStrided));
      return dst.add_(self);
        */
}

pub fn copy_sparse(
    self_:        &mut SparseTensor,
    src:          &SparseTensor,
    non_blocking: bool) -> &mut SparseTensor {
    
    todo!();
        /*
            if (is_same_tensor(self, src))
        return self;
      get_sparse_impl(self)->resize_(
          src.sparse_dim(), src.dense_dim(), src.sizes());
      copy_into_sparse(self, src._indices(), src._values(), non_blocking);
      return self._coalesced_(src.is_coalesced());
        */
}

impl Coalesce for SparseTensor {

    fn coalesce(&self) -> SparseTensor {
        
        todo!();
            /*
                // See NOTE: [ coalesce autograd ]
          if (self.is_coalesced()) {
            return self;
          }
          return _coalesce(self);
            */
    }
}

impl CoalesceSparseCpu {

    fn coalesce_sparse_cpu(&self) -> SparseTensor {
        
        todo!();
            /*
                AT_ASSERT(self.defined());
          TORCH_INTERNAL_ASSERT(variable_excluded_from_dispatch());
          AT_ASSERT(self.is_sparse());
          TORCH_INTERNAL_ASSERT(!self.is_coalesced());

          // NOTE: Since `coalesce` is not an in-place operation when `is_coalesced` is false,
          // we should keep the original tensor intact and do coalesce on a copy of the tensor
          if (self._nnz() < 2) {
            SparseTensor dst = self.clone();
            dst._coalesced_(true);
            return dst;
          }

          Tensor indices = self._indices();
          Tensor values = self._values().contiguous();
          i64 sparse_dim = self.sparse_dim();
          i64 dense_dim = self.dense_dim();
          i64 nnz = self._nnz();

          Tensor indices_scalar = flatten_indices(indices, self.sizes());

          SparseTensor dst = new_sparse(
              optTypeMetaToScalarType(self.options().dtype_opt()),
              self.options().layout_opt(),
              self.options().device_opt(),
              self.options().pinned_memory_opt());
          get_sparse_impl(dst)->resize_(sparse_dim, dense_dim, self.sizes());
          // TODO: is there a more idiomatic way to do this?
          Tensor newIndices = empty(indices.sizes(), indices.options());
          Tensor newValues = empty(values.sizes(), values.options());
          alias_into_sparse(dst, newIndices, newValues);

          Tensor indicesBuffer;
          Tensor indicesPermutation;
          tie(indicesBuffer, indicesPermutation) = indices_scalar.sort(0);
          // NB: The accessor accesses here rely on self._nnz() > 0 (tested earlier in
          // this function)
          auto newIndicesAccessor = newIndices.accessor<i64, 2>();
          auto indicesAccessor = indices.accessor<i64, 2>();
          auto indicesPermutationAccessor = indicesPermutation.accessor<i64, 1>();
          auto indicesBufferAccessor = indicesBuffer.accessor<i64, 1>();

          i64 i = -1;
          AT_DISPATCH_ALL_TYPES_AND_COMPLEX(values.scalar_type(), "coalesce", [&] {
            i64 prev = -1;
            i64 blockSize = values.stride(0);
            Scalar* values_ptr = values.data_ptr<Scalar>();
            Scalar* newValues_ptr = newValues.data_ptr<Scalar>();
            for (i64 j = 0; j < nnz; j++) {
              i64 pos = indicesPermutationAccessor[j];
              i64 curr = indicesBufferAccessor[j];
              if (curr == prev) {
                if (values.numel() >
                    0) { // if values is an empty tensor, there are no elements to copy
                  native::cpublas::axpy<Scalar>(
                      blockSize,
                      1,
                      values_ptr + pos * blockSize,
                      1,
                      newValues_ptr + i * blockSize,
                      1);
                }
              } else {
                ++i;
                for (i64 d = 0; d < sparse_dim; d++) {
                  newIndicesAccessor[d][i] = indicesAccessor[d][pos];
                }
                if (values.numel() >
                    0) { // if values is an empty tensor, there are no elements to copy
                  native::cpublas::copy<Scalar>(
                      blockSize,
                      values_ptr + pos * blockSize,
                      1,
                      newValues_ptr + i * blockSize,
                      1);
                }
              }
              prev = curr;
            }
          });

          dst._coalesced_(true);
          get_sparse_impl(dst)->set_nnz_and_narrow(i + 1);

          return dst;
            */
    }
}

/**
  | sparse_mask(D, S) -> S
  |
  | Filter Tensor D by S.indices() and output
  | a SparseTensor.
  |
  | D and S must share the same shape.
  */
#[inline] pub fn sparse_mask_out_cpu_kernel<Scalar>(
    r_values:     &mut Tensor,
    t:            &Tensor,
    r_nnz:        i64,
    sparse_dim:   i64,
    mask_indices: &Tensor)  {

    todo!();
    /*
       auto r_values_accessor = r_values.accessor<Scalar, 1>();
      auto mask_indices_accessor = mask_indices.accessor<i64, 2>();
      Scalar* t_ptr = t.data_ptr<Scalar>();

      parallel_for(0, r_nnz, 1000, [&](i64 start, i64 end) {
        for (auto i = start; i < end; i++) {
          i64 idx = 0;
          for (i64 d = 0; d < sparse_dim; d++) {
            idx += mask_indices_accessor[d][i] * t.stride(d);
          }
          r_values_accessor[i] = t_ptr[idx];
        }
      });
        */
}

pub fn sparse_mask_out_cpu(
    r:    &mut SparseTensor,
    t:    &Tensor,
    mask: &SparseTensor) -> &mut SparseTensor {
    
    todo!();
        /*
            TORCH_CHECK(mask.is_coalesced(), "sparse_mask: mask is uncoalesced");
      TORCH_CHECK(
          mask.sizes().equals(t.sizes()),
          "sparse_mask: operands have incompatible sizes; self has size ",
          t.sizes(),
          " but mask has size ",
          mask.sizes());
      AT_ASSERT(!t.is_cuda()); // we were supposed to have dispatched on this
      TORCH_CHECK(
          !r.is_cuda(), "sparse_mask: expected 'out' to be CPU, but got CUDA");
      TORCH_CHECK(
          !mask.is_cuda(), "sparse_mask: expected 'mask' to be CPU, but got CUDA");
      resize_as_sparse_(r, mask);
      if (mask._nnz() == 0) {
        return r.zero_();
      }
      i64 dim = t.dim();
      i64 sparse_dim = mask.sparse_dim();
      Tensor mask_indices = mask._indices();
      Tensor mask_values = mask._values();
      Tensor r_values = empty(mask_values.sizes(), r._values().options());
      alias_into_sparse(r, mask_indices.clone(), r_values);
      r._coalesced_(mask.is_coalesced());
      i64 r_nnz = mask._nnz();
      get_sparse_impl(r)->set_nnz_and_narrow(r_nnz);

      if (t.numel() ==
          0) { // if t is an empty tensor, there is no need to mask its elements
        return r;
      }

      if (dim > sparse_dim) {
        // Get a flattened sparse indices, similar to NOTE [ Flatten Sparse Indices
        // ]. Keeping this implementation because it is faster than
        // flatten_indices()
        Tensor indices = zeros({mask._nnz()}, mask_indices.options());
        for (i64 d = 0; d < mask.sparse_dim(); d++) {
          indices.mul_(mask.size(d));
          indices.add_(mask_indices.select(0, d));
        }

        vector<i64> view_size(1 + mask.dense_dim());
        view_size[0] = -1;
        for (i64 d = 0; d < mask.dense_dim(); d++) {
          view_size[d + 1] = mask.size(mask.sparse_dim() + d);
        }

        Tensor t_view = t.view(view_size);
        // TODO: Re-audit this; it used to be an indexSelect directly into r_values
        index_select_out(r_values, t_view, 0, indices);
      } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX(r_values.scalar_type(), "sparse_mask", [&] {
          sparse_mask_out_cpu_kernel<Scalar>(
              r_values, t, r_nnz, sparse_dim, mask_indices);
        });
      }
      return r;
        */
}

pub fn sparse_mask_cpu(
    t:    &Tensor,
    mask: &SparseTensor) -> SparseTensor {
    
    todo!();
        /*
            SparseTensor r = empty({0}, t.options().layout(kSparse));
      sparse_mask_out_cpu(r, t, mask);
      return r;
        */
}

pub fn sparse_mask_helper_cpu(
    t:            &SparseTensor,
    mask_indices: &Tensor) -> Tensor {
    
    todo!();
        /*
            /*
        This is a helper function which filter values from `t._values()` using the
        `mask_indices`. This CPU implementation uses a simple hash_map to filter
        values by matching the `mask_indices` with the indices at tensor input `t`.

        Inputs:
          `t`             - coalesced sparse tensor input
          `mask_indices`  - mask indices tensor

        Note: The nnz in the output tensor will be same as the `mask_indices`. So it
        will works independently if the mask is coalesced or not.
      */
      TORCH_CHECK(t.is_sparse(), "t: input is not a sparse tensor");
      TORCH_CHECK(t.is_coalesced(), "t:  input is uncoalesced");
      TORCH_CHECK(
          mask_indices.dim() == t._indices().dim(),
          "mask_indices: operands have incompatible indices dim; self has dim ",
          t._indices().dim(),
          " but mask has dim ",
          mask_indices.dim());
      TORCH_CHECK(
          mask_indices.is_contiguous(), "mask_indices: mask is not contiguous");

      i64 r_nnz = mask_indices.size(1);
      auto t_v = t._values();
      auto vsize = t_v.sizes().vec();
      vsize[0] = r_nnz;

      Tensor r_values = zeros(vsize, t_v.options());
      auto t_i = t._indices();
      auto t_nnz = t._nnz();

      unordered_map<i64, i64> t_flatten_indices =
          unordered_map<i64, i64>{};
      auto full_size = t.sizes();
      auto ti_flattened_indices = sparse::flatten_indices(t_i, full_size);

      // Step 1: flatten the sparse indices `t._indices()` tensor and then  map this
      // flatten value `index` to the original position `i`
      for (i64 i = 0; i < t_nnz; i++) {
        i64 index = ti_flattened_indices.data_ptr<i64>()[i];
        t_flatten_indices[index] = i;
      }

      // Step 2: Filter `t._values()` values by matching the flatten `mask_indices`
      // with the flatten `t._indices()` using the hash_map `t_flatten_indices`

      auto flattened_mask_indices =
          sparse::flatten_indices(mask_indices, full_size);
      parallel_for(0, r_nnz, 0, [&](i64 start, i64 end) {
        for (auto i = start; i < end; i++) {
          i64 index = flattened_mask_indices.data_ptr<i64>()[i];
          auto iter = t_flatten_indices.find(index);
          if (iter != t_flatten_indices.end()) {
            r_values[i] = t_v[iter->second];
          }
        }
      });
      return r_values;
        */
}
