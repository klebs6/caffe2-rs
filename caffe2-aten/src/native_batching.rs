crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Batching.cpp]

/**
  | Adds a batch dimension to the tensor
  | `self` out-of-place
  |
  */
pub fn add_batch_dim(
        self_:     &Tensor,
        batch_dim: i64,
        level:     i64) -> Tensor {
    
    todo!();
        /*
            return addBatchDim(self, level, batch_dim);
        */
}

pub fn has_level(
        self_: &Tensor,
        level: i64) -> bool {
    
    todo!();
        /*
            const auto* batched = maybeGetBatchedImpl(self);
      if (!batched) {
        return false;
      }
      auto bdims = batched->bdims();
      auto* it = find_if(bdims.begin(), bdims.end(), [&](const BatchDim& bdim) {
        return bdim.level() == level;
      });
      return it != bdims.end();
        */
}

/**
  | Returns a Tensor with batch dim with level
  | `level` turned into a regular dimension, as
  | well as a logical dim index of where said
  | dimension is in the returned tensor.
  |
  | A call to this function is always followed by
  | a call to `movedim`.
  |
  | Preconditions: A BatchDim with level `level`
  | must exist inside `batched`.
  |
  | The reason why we want to return the index of
  | where said dimension is in the returned tensor
  | is because we want to keep track of which
  | dimension used to be the batch dimension so
  | that we can move it to the correct logical
  | dimension specified by `out_dims` in vmap. For
  | example, if we had
  |
  | >>> x = torch.randn(2, 3, 5)
  | >>> vmap(lambda x: x, in_dims=0, out_dims=1)(x)
  |
  | then right when we are about to exit the vmap
  | block, x is a BatchedTensor with a batch
  | dimension at (physical) index 0. Note that the
  | batch dimension doesn't always have to exist at
  | (physical) index 0. When we undo the batch
  | dimension, we want to move it to dimension
  | 1 (as specified by out_dims). So we return the
  | index at which the batch dim appears so that we
  | can move it to the correct place. later down
  | the line via a call to `movedim`.
  */
pub fn remove_existing_batch_dim(
        batched: *const BatchedTensorImpl,
        level:   i64) -> (Tensor,i64) {
    
    todo!();
        /*
            auto bdims = batched->bdims();
      if (bdims.size() == 1) {
        TORCH_INTERNAL_ASSERT(bdims[0].level() == level);
        return make_pair(batched->value(), bdims[0].dim());
      }
      BatchDims new_bdims;
      i64 newly_exposed_physical_dim = -1;
      new_bdims.reserve(bdims.size() - 1);
      for (const auto& bdim : bdims) {
        if (bdim.level() == level) {
          newly_exposed_physical_dim = bdim.dim();
        } else {
          new_bdims.push_back(bdim);
        }
      }
      // Because a BatchDim with level `level` must exist inside `batched,
      // we should have found a `newly_exposed_logical_dim`.
      TORCH_INTERNAL_ASSERT(newly_exposed_physical_dim != -1);
      i64 num_batch_dims_before_newly_exposed_physical_dim = count_if(
          new_bdims.begin(), new_bdims.end(),
          [&](const BatchDim& bdim) {
            return bdim.dim() < newly_exposed_physical_dim;
          });
      i64 newly_exposed_logical_dim =
          newly_exposed_physical_dim - num_batch_dims_before_newly_exposed_physical_dim;
      auto result_tensor = makeBatched(batched->value(), move(new_bdims));
      return make_pair(move(result_tensor), newly_exposed_logical_dim);
        */
}

/**
  | Poor man's version of np.moveaxis. Moves the
  | dimension at `dst` to `src` while preserving
  | the order of other existing dimensions.
  |
  | We should probably add np.moveaxis (it is more
  | general) to PyTorch. (#36048)
  |
  | When we do, replace the following with it.
  */
pub fn movedim(
        self_: &Tensor,
        src:   i64,
        dst:   i64) -> Tensor {
    
    todo!();
        /*
            auto logical_dim = self.dim();
      src = maybe_wrap_dim(src, logical_dim);
      dst = maybe_wrap_dim(dst, logical_dim);
      if (src == dst) {
        return self;
      }
      VmapDimVector permutation;
      permutation.reserve(logical_dim);
      for (i64 dim = 0; dim < logical_dim; dim++) {
        if (dim == src) {
          continue;
        }
        permutation.push_back(dim);
      }
      permutation.insert(permutation.begin() + dst, src);
      return self.permute(permutation);
        */
}

/**
  | Removes the batch dim with level `level` from
  | `self`. If this causes the last batch dim to be
  | removed from a BatchedTensor, then this returns
  | a regular Tensor.
  |
  | If the `level` of the batch dim to remove does
  | not exist in `self`, then we add the batch dim
  | in. This can happen if `self` didn't interact
  | with a tensor
  |
  | inside the vmap level, for example,
  |     self = torch.randn(3)
  |     y = torch.randn(5)
  |     out = vmap(lambda x: vmap(lambda y: x)(y))(self)
  |     assert out.shape == (3, 5)
  |
  | Inside the inner vmap, `x` is a BatchedTensor
  | with a single batch dimension corresponding to
  | the *outer* vmap level and it doesn't have any
  | dimensions that correspond to the inner vmap
  | level so we need to create one for the user.
  |
  | `out_dim` controls where we should put the
  | batch dimension in the output tensor.
  |
  */
pub fn remove_batch_dim(
        self_:      &Tensor,
        level:      i64,
        batch_size: i64,
        out_dim:    i64) -> Tensor {
    
    todo!();
        /*
            if (!has_level(self, level)) {
        auto self_sizes = self.sizes();
        VmapDimVector expanded_sizes(self_sizes.begin(), self_sizes.end());
        expanded_sizes.insert(expanded_sizes.begin() + out_dim, batch_size);
        return self.expand(expanded_sizes);
      }

      // Must be batched if has_level(self, /*any_level*/)
      const auto* batched = maybeGetBatchedImpl(self);
      TORCH_INTERNAL_ASSERT(batched != nullptr);

      Tensor self_without_bdim;
      i64 newly_exposed_logical_dim;
      tie(self_without_bdim, newly_exposed_logical_dim) = remove_existing_batch_dim(batched, level);
      return movedim(self_without_bdim, newly_exposed_logical_dim, out_dim);
        */
}
