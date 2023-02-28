crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/SparseTensorImpl.h]

/// Stored in COO format, indices + values.
///
pub struct SparseTensorImpl {
    base: TensorImpl,

    // INVARIANTS:
    //
    // sparse_dim: range [0, len(shape)]; sparse_dim
    // + dense_dim = len(shape)
    //
    // dense_dim : range [0, len(shape)]; sparse_dim
    // + dense_dim = len(shape)
    //
    // _indices.shape: dimensionality: 2,  shape:
    // (sparse_dim, nnz)
    //
    // _values.shape:  dimensionality:
    // 1 + dense_dim.  shape: (nnz,
    // shape[sparse_dim:])

    /**
      | number of sparse dimensions
      |
      */
    sparse_dim: i64, // default = 0

    /**
      | number of dense dimensions
      |
      */
    dense_dim:  i64, // default = 0

    /**
      | always a LongTensor
      |
      */
    indices:    Tensor,

    values:     Tensor,

    /**
      | A sparse tensor is 'coalesced' if every
      | index occurs at most once in the indices
      | tensor, and the indices are in sorted
      | order. (This means that it is very easy
      | to convert a coalesced tensor to CSR
      | format: you need only compute CSR format
      | indices.)
      | 
      | Most math operations can only be performed
      | on coalesced sparse tensors, because
      | many algorithms proceed by merging
      | two sorted lists (of indices).
      |
      */
    coalesced:  bool, // default = false
}

impl SparseTensorImpl {

    /**
      | compute_numel with integer multiplication
      | overflow check, see gh-57542
      |
      */
    pub fn refresh_numel(&mut self)  {
        
        todo!();
        /*
            TensorImpl::safe_refresh_numel();
        */
    }

    pub fn nnz(&self) -> i64 {
        
        todo!();
        /*
            return values_.size(0);
        */
    }
    
    pub fn sparse_dim(&self) -> i64 {
        
        todo!();
        /*
            return sparse_dim_;
        */
    }
    
    pub fn dense_dim(&self) -> i64 {
        
        todo!();
        /*
            return dense_dim_;
        */
    }
    
    pub fn coalesced(&self) -> bool {
        
        todo!();
        /*
            return coalesced_;
        */
    }
    
    pub fn indices(&self) -> Tensor {
        
        todo!();
        /*
            return indices_;
        */
    }
    
    pub fn values(&self) -> Tensor {
        
        todo!();
        /*
            return values_;
        */
    }
    
    /**
      | WARNING: This function does NOT preserve
      | invariants of sparse_dim/dense_dim
      | with respect to indices and values
      |
      */
    pub fn raw_resize(&mut self, 
        sparse_dim: i64,
        dense_dim:  i64,
        size:       &[i32])  {
        
        todo!();
        /*
            TORCH_CHECK(allow_tensor_metadata_change(), "raw_resize_ ", err_msg_tensor_metadata_change_not_allowed);
        sizes_and_strides_.set_sizes(size);
        sparse_dim_ = sparse_dim;
        dense_dim_ = dense_dim;
        refresh_numel();
        */
    }

    /**
      | NOTE: This function preserves invariants of
      | sparse_dim/dense_dim with respect to indices
      | and values.
      |
      | NOTE: This function supports the following
      | cases:
      |
      | 1. When we keep the number of dense
      | dimensions unchanged, and NOT shrinking the
      | size of any of the dense dimensions.
      |
      | 2. When we keep the number of sparse
      | dimensions unchanged, and NOT shrinking the
      | size of any of the sparse dimensions.
      |
      | 3. When the sparse tensor has zero nnz, in
      | which case we are free to change the shapes
      | of both its sparse and dense dimensions.
      |
      | This function DOESN'T support (and will throw
      | an error) the following cases:
      |
      | 1. When we attempt to change the number of
      | sparse dimensions on a non-empty sparse
      | tensor (such an operation will invalidate the
      | indices stored).
      |
      | 2. When we attempt to change the number of
      | dense dimensions on a non-empty sparse tensor
      | (such an operation will behave differently
      | from an equivalent dense tensor's resize
      | method, and for API consistency we don't
      | support it).
      |
      | 3. When we attempt to shrink the size of any
      | of the dense dimensions on a non-empty sparse
      | tensor (such an operation will behave
      | differently from an equivalent dense tensor's
      | resize method, and for API consistency we
      | don't support it).
      |
      | 4. When we attempt to shrink the size of any
      | of the sparse dimensions on a non-empty
      | sparse tensor (this could make some of the
      | stored indices out-of-bound and thus unsafe).
      |
      */
    pub fn resize(&mut self, 
        sparse_dim: i64,
        dense_dim:  i64,
        size:       &[i32])  {
        
        todo!();
        /*
            TORCH_CHECK(allow_tensor_metadata_change(), "resize_ ", err_msg_tensor_metadata_change_not_allowed);
        TORCH_CHECK(sparse_dim + dense_dim == static_cast<i64>(size.size()), "number of dimensions must be sparse_dim (", sparse_dim, ") + dense_dim (", dense_dim, "), but got ", size.size());
        if (nnz() > 0) {
          auto alt_options_msg = "You could try the following options:\n\
    1. If you need an empty sparse tensor of this size, call `x = torch.sparse_coo_tensor(size)`.\n\
    2. If you need to resize this tensor, you have the following options:\n\
        1. For both sparse and dense dimensions, keep the number of them constant and the size of them non-shrinking, and then try the same call again.\n\
        2. Or, create a new sparse tensor with the correct indices and values from this sparse tensor.";

          TORCH_CHECK(sparse_dim == sparse_dim_,
            "changing the number of sparse dimensions (from ", sparse_dim_, " to ", sparse_dim, ") on a non-empty sparse tensor is not supported.\n", alt_options_msg);

          TORCH_CHECK(dense_dim == dense_dim_,
            "changing the number of dense dimensions (from ", dense_dim_, " to ", dense_dim, ") on a non-empty sparse tensor is not supported.\n", alt_options_msg);

          bool shrinking_sparse_dims = false;
          bool shrinking_dense_dim = false;
          auto sparse_size_original = sizes().slice(0, sparse_dim);
          auto sparse_size_new = size.slice(0, sparse_dim);
          for (i64 i = 0; i < sparse_dim; i++) {
            if (sparse_size_new[i] < sparse_size_original[i]) {
              shrinking_sparse_dims = true;
              break;
            }
          }
          auto dense_size_original = sizes().slice(sparse_dim);
          auto dense_size_new = size.slice(sparse_dim);
          for (i64 i = 0; i < dense_dim; i++) {
            if (dense_size_new[i] < dense_size_original[i]) {
              shrinking_dense_dim = true;
              break;
            }
          }

          TORCH_CHECK(!shrinking_sparse_dims,
            "shrinking the size of sparse dimensions (from ", sparse_size_original, " to ", sparse_size_new, ") on a non-empty sparse tensor is not supported.\n", alt_options_msg);

          TORCH_CHECK(!shrinking_dense_dim,
            "shrinking the size of dense dimensions (from ", dense_size_original, " to ", dense_size_new, ") on a non-empty sparse tensor is not supported.\n", alt_options_msg);
        }

        const bool size_equals_sizes = equal(size.begin(), size.end(), sizes_and_strides_.sizes_begin(), sizes_and_strides_.sizes_end());
        if ((!size_equals_sizes) || (sparse_dim != sparse_dim_) || (dense_dim != dense_dim_)) {
          auto nnz = values().size(0);
          vector<i64> values_size = {nnz};
          auto dense_size = size.slice(sparse_dim);
          values_size.insert(values_size.end(), dense_size.begin(), dense_size.end());
          values_.resize_(values_size);
          indices_.resize_({sparse_dim, nnz});
        }

        if (!size_equals_sizes) {
          sizes_and_strides_.set_sizes(size);
        }
        sparse_dim_ = sparse_dim;
        dense_dim_ = dense_dim;
        refresh_numel();
        */
    }

    /**
      | NOTE: this function will resize the sparse
      | tensor and also set `indices` and `values`
      | to empty.
      |
      */
    pub fn resize_and_clear(&mut self, 
        sparse_dim: i64,
        dense_dim:  i64,
        size:       &[i32])  {
        
        todo!();
        /*
            TORCH_CHECK(allow_tensor_metadata_change(), "resize_and_clear_ ", err_msg_tensor_metadata_change_not_allowed);
        TORCH_CHECK(sparse_dim + dense_dim == static_cast<i64>(size.size()), "number of dimensions must be sparse_dim (", sparse_dim, ") + dense_dim (", dense_dim, "), but got ", size.size());

        sizes_and_strides_.set_sizes(size);
        sparse_dim_ = sparse_dim;
        dense_dim_ = dense_dim;

        auto empty_indices = empty({sparse_dim, 0}, indices().options());
        vector<i64> values_size = {0};
        auto dense_size = sizes().slice(sparse_dim);
        values_size.insert(values_size.end(), dense_size.begin(), dense_size.end());
        auto empty_values = empty(values_size, values().options());
        set_indices_and_values_unsafe(empty_indices, empty_values);
        refresh_numel();
        */
    }
    
    pub fn set_coalesced(&mut self, coalesced: bool)  {
        
        todo!();
        /*
            TORCH_CHECK(allow_tensor_metadata_change(), "set_coalesced ", err_msg_tensor_metadata_change_not_allowed);
        coalesced_ = coalesced;
        */
    }

    /**
      | -----------
      | @note
      | 
      | this function is only used internally
      | and not exposed to Python frontend
      |
      */
    pub fn set_nnz_and_narrow(&mut self, new_nnz: i64)  {
        
        todo!();
        /*
            TORCH_CHECK(allow_tensor_metadata_change(), "set_nnz_and_narrow ", err_msg_tensor_metadata_change_not_allowed);
        AT_ASSERT(new_nnz <= nnz());
        indices_ = indices_.narrow(1, 0, new_nnz);
        values_ = values_.narrow(0, 0, new_nnz);
        */
    }

    /**
      | Return a TensorImpl that is a shallow-copy
      | of this TensorImpl.
      | 
      | For usage of `version_counter` and
      | `allow_tensor_metadata_change`,
      | see NOTE [ TensorImpl Shallow-Copying
      | ].
      |
      */
    pub fn shallow_copy_and_detach(&self, 
        version_counter:              &VariableVersion,
        allow_tensor_metadata_change: bool) -> IntrusivePtr<TensorImpl> {
        
        todo!();
        /*
            auto impl = make_intrusive<SparseTensorImpl>(key_set(), dtype());
        copy_tensor_metadata(
          /*src_impl=*/this,
          /*dest_impl=*/impl.get(),
          /*version_counter=*/version_counter,
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
        impl->refresh_numel();
        return impl;
        */
    }

    /**
      | Return a TensorImpl that is a shallow-copy
      | of this TensorImpl.
      | 
      | For usage of `version_counter` and
      | `allow_tensor_metadata_change`,
      | see NOTE [ TensorImpl Shallow-Copying ].
      |
      */
    pub fn shallow_copy_and_detach(&self, 
        version_counter:              VariableVersion,
        allow_tensor_metadata_change: bool) -> IntrusivePtr<TensorImpl> {
        
        todo!();
        /*
            auto impl = make_intrusive<SparseTensorImpl>(key_set(), dtype());
        copy_tensor_metadata(
          /*src_impl=*/this,
          /*dest_impl=*/impl.get(),
          /*version_counter=*/move(version_counter),
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
        impl->refresh_numel();
        return impl;
        */
    }

    /**
      | Shallow-copies data from another TensorImpl
      | into this TensorImpl.
      | 
      | For why this function doesn't check
      | this TensorImpl's `allow_tensor_metadata_change_`,
      | see NOTE [ TensorImpl Shallow-Copying
      | ].
      |
      */
    pub fn shallow_copy_from(&mut self, impl_: &IntrusivePtr<TensorImpl>)  {
        
        todo!();
        /*
            AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
        auto sparse_impl = static_cast<const SparseTensorImpl*>(impl.get());
        copy_tensor_metadata(
          /*src_impl=*/sparse_impl,
          /*dest_impl=*/this,
          /*version_counter=*/version_counter(),
          /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
        refresh_numel();
        */
    }
    
    /**
      | Copy the tensor metadata fields (e.g.
      | sizes / strides / storage pointer / storage_offset)
      | from one TensorImpl to another TensorImpl.
      | 
      | For usage of `version_counter` and
      | `allow_tensor_metadata_change`,
      | see NOTE [ TensorImpl Shallow-Copying
      | ].
      |
      */
    pub fn copy_tensor_metadata(
        src_sparse_impl:              *const SparseTensorImpl,
        dest_sparse_impl:             *mut SparseTensorImpl,
        version_counter:              &VariableVersion,
        allow_tensor_metadata_change: bool)  {
        
        todo!();
        /*
            TensorImpl::copy_tensor_metadata(src_sparse_impl, dest_sparse_impl, version_counter, allow_tensor_metadata_change);

        // Sparse-specific fields
        dest_sparse_impl->sparse_dim_ = src_sparse_impl->sparse_dim();
        dest_sparse_impl->dense_dim_ = src_sparse_impl->dense_dim();
        dest_sparse_impl->indices_ = src_sparse_impl->indices();
        dest_sparse_impl->values_ = src_sparse_impl->values();
        dest_sparse_impl->coalesced_ = src_sparse_impl->coalesced();
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/SparseTensorImpl.cpp]

pub fn sparse_tensor_set_to_device_type(key_set: DispatchKeySet) -> DeviceType {
    
    todo!();
        /*
            if (key_set.has(DispatchKey::SparseCPU)) {
          return kCPU;
        } else if (key_set.has(DispatchKey::SparseXPU)) {
          return kXPU;
        } else if (key_set.has(DispatchKey::SparseCUDA)) {
          return kCUDA;
        } else {
          AT_ERROR("Cannot construct SparseTensor with non-sparse tensor type ID ", key_set);
        }
        */
}

impl SparseTensorImpl {
    
    /**
      | An empty dense tensor defaults to
      | a 1-dimensional tensor of size [0] (recall, it
      | is not a 0-dimensional tensor, because such
      | a tensor would a scalar and have one element)
      |
      | Thus, an empty sparse tensor should be
      | a 1-dimensional tensor of size [0].
      |
      | Furthermore, we have dim == sparse_dim
      | + dense_dim; since this is a sparse tensor, let
      | us say that an empty sparse tensor has
      | sparse_dim == 1 and dense_dim == 0.  (There is
      | a degree of freedom here, but given that this
      | is a sparse dimension, it seems reasonable to
      | demand that sparse_dim > 0).
      |
      | This means that we allocate a [1,0] size
      | indices tensor and a [0] size values tensor for
      | such an empty tensor.
      */
    pub fn new(
        key_set:   DispatchKeySet,
        data_type: TypeMeta) -> Self {
    
        todo!();
        /*


            :   SparseTensorImpl(key_set, data_type
          , empty({1, 0}, initialTensorOptions().device(sparseTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
          , empty({0}, initialTensorOptions().device(sparseTensorSetToDeviceType(key_set)).dtype(data_type)))
        */
    }
    
    pub fn new(
        key_set:   DispatchKeySet,
        data_type: TypeMeta,
        indices:   Tensor,
        values:    Tensor) -> Self {
    
        todo!();
        /*


            : TensorImpl(key_set, data_type, values.device())
        , sparse_dim_(1)
        , dense_dim_(0)
        , indices_(move(indices))
        , values_(move(values)) 
      // we proxy to this constructor so we can initialize the device correctly, but really only indices/values of this shape are allowed.
      AT_ASSERT(indices_.sizes() == IntArrayRef({1, 0}));
      AT_ASSERT(values_.sizes() == IntArrayRef({0}));
      AT_ASSERT(values_.device() == indices_.device());
      AT_ASSERT(values_.device() == device());

      is_non_overlapping_and_dense_ = false;
      set_storage_access_should_throw();
      set_has_contiguity_policy(HasContiguityPolicy::ContiguityNotSupported);
        */
    }
    
    pub fn release_resources(&mut self)  {
        
        todo!();
        /*
            TensorImpl::release_resources();
      values_.reset();
      indices_.reset();
        */
    }
    
    pub fn strides(&self) -> &[i32] {
        
        todo!();
        /*
            AT_ERROR("sparse tensors do not have strides");
        */
    }
    
    pub fn stride(&self, d: i64) -> i64 {
        
        todo!();
        /*
            AT_ERROR("sparse tensors do not have strides");
        */
    }
    
    pub fn set_size(&mut self, 
        dim:      i64,
        new_size: i64)  {
        
        todo!();
        /*
            AT_ERROR("sparse tensors do not have set_size");
        */
    }
    
    pub fn set_stride(&mut self, 
        dim:        i64,
        new_stride: i64)  {
        
        todo!();
        /*
            AT_ERROR("sparse tensors do not have set_stride");
        */
    }
    
    pub fn set_storage_offset(&mut self, storage_offset: i64)  {
        
        todo!();
        /*
            AT_ERROR("sparse tensors do not have set_storage_offset");
        */
    }

    #[cfg(debug_assertions)]
    pub fn has_storage(&self) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!storage_, "SparseTensorImpl assumes that storage_ is never set");
      return false;
        */
    }
    
    pub fn tensorimpl_type_name(&self) -> *const u8 {
        
        todo!();
        /*
            return "SparseTensorImpl";
        */
    }
    
    /**
      | Takes indices and values and directly puts
      | them into the sparse tensor, no copy.
      |
      | NOTE: this function is unsafe because it
      | doesn't check whether any indices are out of
      | boundaries of `sizes`, so it should ONLY be
      | used where we know that the indices are
      | guaranteed to be within bounds.
      |
      | This used to be called THSTensor_(_move)
      |
      | NB: This used to be able to avoid a refcount
      | bump, but I was too lazy to make it happen
      |
      */
    pub fn set_indices_and_values_unsafe(&mut self, 
        indices: &Tensor,
        values:  &Tensor)  {
        
        todo!();
        /*
            TORCH_CHECK(allow_tensor_metadata_change(), "set_indices_and_values_unsafe ", err_msg_tensor_metadata_change_not_allowed);

      TORCH_CHECK(!indices.is_sparse(), "expected indices to be a dense tensor, but got indices of layout ", indices.layout());
      TORCH_CHECK(!values.is_sparse(), "expected values to be a dense tensor, but got values of layout ", values.layout());

      TORCH_CHECK(values.device().type() == device().type(), "device type of values (", values.device().type(), ") must match device type of device().type()", device().type(), ")");
      TORCH_CHECK(values.scalar_type() == typeMetaToScalarType(dtype()), "dtype of values (", values.scalar_type(), ") must match dtype of sparse tensor (", typeMetaToScalarType(dtype()), ")");
      TORCH_CHECK(indices.scalar_type() == kLong, "indices must be an int64 tensor");
      TORCH_CHECK(indices.options().backend() == values.options().backend(), "backend of indices (", indices.options().backend(), ") must match backend of values (", values.options().backend(), ")");
      TORCH_CHECK(!indices.is_cuda() || indices.get_device() == values.get_device(), "device of indices (", indices.get_device(), ") must match device of values (", values.get_device(), ")");

      TORCH_CHECK(indices.dim() == 2, "indices must be sparse_dim x nnz, but got: ", indices.sizes());
      TORCH_CHECK(indices.size(1) == values.size(0), "indices and values must have same nnz, but got nnz from indices: ", indices.size(1), ", nnz from values: ", values.size(0));
      TORCH_CHECK(indices.size(0) == sparse_dim_, "indices has incorrect first dimension, expected ", sparse_dim_, ", got ", indices.size(0));
      TORCH_CHECK(values.dim() == dense_dim_ + 1, "values has incorrect number of dimensions, expected ", dense_dim_ + 1, ", got ", values.dim());

      auto dense_size_original = sizes().slice(sparse_dim_);
      vector<i64> expected_values_size_vec = {values.size(0)};
      expected_values_size_vec.insert(expected_values_size_vec.end(), dense_size_original.begin(), dense_size_original.end());
      IntArrayRef expected_values_size(expected_values_size_vec);
      auto new_values_size = values.sizes();
      TORCH_CHECK(
        equal(expected_values_size.begin(), expected_values_size.end(), new_values_size.begin()),
        "values has incorrect size, expected ", expected_values_size, ", got ", new_values_size
      );

      indices_ = indices;
      values_ = values;
      AT_ASSERT(device() == values_.device());
      AT_ASSERT(values_.device() == indices_.device());

      coalesced_ = false;
        */
    }
}
