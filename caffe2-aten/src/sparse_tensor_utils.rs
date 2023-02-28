crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/SparseTensorUtils.h]

pub type SparseTensor = Tensor;
pub type SparseType   = Type;

/**
  | This is an internal utility function for
  | getting at the SparseTensorImpl, so that we can
  | write sparse tensor specific accessors for
  | special fields in SparseTensor.
  |
  | You should only use this for writing low level
  | setters/getters for SparseTensorImpl fields;
  | otherwise, you should use the low level
  | setters/getters that were implemented using
  | this.
  |
  | This may be called repeatedly, so make sure
  | it's pretty cheap.
  */
#[inline] pub fn get_sparse_impl(self_: &SparseTensor) -> *mut SparseTensorImpl {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(self.is_sparse(), "_internal_get_SparseTensorImpl: not a sparse tensor");
      return static_cast<SparseTensorImpl*>(self.unsafeGetTensorImpl());
        */
}

/**
  | Takes indices and values and directly puts them
  | into the sparse tensor, no copy.
  |
  | This used to be called THSTensor_(_move)
  |
  */
#[inline] pub fn alias_into_sparse(
        self_:   &SparseTensor,
        indices: &Tensor,
        values:  &Tensor)  {
    
    todo!();
        /*
            get_sparse_impl(self)->set_indices_and_values_unsafe(indices, values);
        */
}

/**
  | Take indices and values and makes a (data) copy
  | of them to put into the sparse indices/values.
  |
  | This used to be called THSTensor_(_set)
  |
  */
#[inline] pub fn copy_into_sparse(
    self_:        &SparseTensor,
    indices:      &Tensor,
    values:       &Tensor,
    non_blocking: bool)  {
    
    todo!();
        /*
            alias_into_sparse(
          self,
          indices.to(self._indices().options(), non_blocking, /*copy=*/true),
          values.to(self._values().options(), non_blocking, /*copy=*/true));
        */
}

/**
  | TODO: put this into the public API
  |
  */
#[inline] pub fn is_same_tensor(
    lhs: &Tensor,
    rhs: &Tensor) -> bool {

    todo!();
        /*
            return lhs.unsafeGetTensorImpl() == rhs.unsafeGetTensorImpl();
        */
}

#[inline] pub fn is_same_density(
    self_: &SparseTensor,
    src:   &SparseTensor) -> bool {
    
    todo!();
        /*
            return self.sparse_dim() == src.sparse_dim() && self.dense_dim() == src.dense_dim();
        */
}

/**
  | Give us a new values tensor, with the same
  | dimensionality as 'values' but with a new
  | number of non-zero elements.
  |
  | TODO: Expose this for real in ATen, some day?
  |
  | NB: Doesn't preserve data.
  */
#[inline] pub fn new_values_with_size_of(
    values: &Tensor,
    nnz:    i64) -> Tensor {
    
    todo!();
        /*
            vector<i64> size = values.sizes().vec();
      size[0] = nnz;
      return empty(size, values.options());
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/SparseTensorUtils.cpp]

/**
  | NOTE [ Flatten Sparse Indices ]
  |
  | This helper function flattens a sparse indices
  | tensor (a Tensor) into a 1D indices
  | tensor. E.g.,
  |
  |   input = [[2, 4, 0],
  |            [3, 1, 10]]
  |   full_size = [2, 12]
  |   output = [ 2 * 12 + 3, 4 * 12 + 1, 0 * 12 + 10 ] = [27, 49, 10]
  |
  | In other words, assuming that each `indices[i,
  | :]` is a valid index to a tensor `t` of shape
  | `full_size`.
  |
  | This returns the corresponding indices to the
  | flattened tensor `t.reshape(
  | prod(full_size[:indices.size(0)]), -1 )`.
  |
  | if forceClone is true, the result will forced
  | to be a clone of self.
  |
  | if force_clone is true, the result will forced
  | to be a clone of self.
  |
  */
pub fn flatten_indices(
        indices:     &Tensor,
        full_size:   &[i32],
        force_clone: bool) -> Tensor {

    let force_clone: bool = force_clone.unwrap_or(false);

    todo!();
        /*
            i64 sparse_dim = indices.size(0);
      if (sparse_dim == 1) {
        if (force_clone) {
          return indices.squeeze(0).clone(MemoryFormat::Contiguous);
        } else {
          return indices.squeeze(0);
        }
      } else {
        vector<i64> indices_mult_cpu_vec;
        indices_mult_cpu_vec.reserve(sparse_dim);
        i64 mult = 1;
        for (i64 i = sparse_dim - 1; i >= 0; i--) {
          indices_mult_cpu_vec[i] = mult;
          mult *= full_size[i];
        }
        auto indices_mult_cpu = from_blob(
            indices_mult_cpu_vec.data(),
            /*size=*/{sparse_dim, 1},
            indices.options().device(kCPU));
        // NB: must be blocking because this blob may be freed after this closure,
        //     and non_blocking copy will see garbage.
        auto indices_mult = indices_mult_cpu.to(indices.device(), /*non_blocking=*/false);
        // Ideally we want matmul but matmul is slow on CPU Long and not implemented
        // on CUDA Long. So mul is faster.
        return indices.mul(indices_mult).sum(0);
      }
        */
}

/**
  | Flatten sparse tensor's indices from nD to 1D,
  | similar to NOTE [ Flatten Sparse Indices ],
  | except this one allows partial flatten: only
  | flatten on specified dims. Note that the
  | flatten indices might be uncoalesced if
  | dims_to_flatten.size() < sparse_dim.
  |
  | Also if input indices is already coalesced, the
  | flattened indices will also be sorted.
  |
  | args:
  |    indices: sparse tensor indices
  |    sizes: sparse tensor sizes
  |    dims_to_flatten: a list of dim index to flatten
  |
  | Ex1:
  |   indices = [[2, 4, 0],
  |             [3, 1, 3]]
  |   sizes = [2, 12]
  |   dims_to_flatten = [0, 1]
  |   new_indices = [ 2 * 12 + 3, 4 * 12 + 1, 0 * 12 + 3 ] = [27, 49, 3]
  |
  | Ex2:
  |   dims_to_flatten = [1]
  |   new_indices = [ 3, 1, 3 ]  # uncoalesced
  */
pub fn flatten_indices_by_dims(
    indices:         &Tensor,
    sizes:           &&[i32],
    dims_to_flatten: &&[i32]) -> Tensor {
    
    todo!();
        /*
            Tensor new_indices = zeros({indices.size(1)}, indices.options());
      for (auto d : dims_to_flatten) {
        new_indices.mul_(sizes[d]);
        new_indices.add_(indices.select(0, d));
      }
      return new_indices;
        */
}

/**
  | Find the CSR representation for a row
  | `indices` from the COO format
  |
  */
pub fn coo_to_csr(
    indices: *const i64,
    dim:     i64,
    nnz:     i64) -> Tensor {
    
    todo!();
        /*
            /*
        Find the CSR representation for a row `indices` from the COO format
        Inputs:
          `indices` is the row pointer from COO indices
          `dim` is the row dimensionality
          `nnz` is the number of non-zeros

        Output:
          `csr` is a compressed row array in a CSR format
      */
      Tensor csr = zeros({dim + 1}, kLong);

      // TODO: eliminate this conditional when zero-size dims supported correctly
      if (nnz > 0) {
        auto csr_accessor = csr.accessor<i64, 1>();
        // Convert the sparse matrix to CSR format
        parallel_for(0, nnz, 10000, [&](i64 start, i64 end) {
          i64 h, hp0, hp1;
          for (auto i = start; i < end; i++) {
            hp0 = indices[i];
            hp1 = (i+1 == nnz) ?  dim : indices[i+1];
            if (hp0 != hp1) {
              for (h = hp0; h < hp1; h++) {
                csr_accessor[h+1] = i+1;
              }
            }
          }
        });
      }
      return csr;
        */
}
