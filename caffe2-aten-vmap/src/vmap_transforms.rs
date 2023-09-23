/*!
  | This file contains abstractions used for
  | transforming *logical* vmap arguments into
  | *physical* arguments. (Keep reading for
  | definitions of these terms).
  |
  | NOTE: [Logical vs physical args]
  |
  | Consider the following vmap. vmap(vmap(func,
  |   in_dims=(2,)), in_dims=(0,))(torch.ones(2, 3,
  |   4))
  |
  | This would produce a BatchedTensor wrapping
  | a Tensor of size [2, 3, 4], with batch dims
  | 0 and 2:
  |
  |   BatchedTensor(ones(2, 3, 4), bdims=[(lvl=1,dim=0),(lvl=2,dim=2)])
  |
  | We say the *logical* view of the tensor has
  | size [3] -- tensors inside `func` appear to
  | have size [3].
  |
  | However, the *physical* underlying tensor (the
  | one passed to vmap) has size [2, 3, 4].
  |
  | This notion of logical vs physical also extends
  | to non-tensor arguments.
  |
  | Consider the previous tensor; let's assume the
  | user called `torch.sum(tensor, dim=0)` inside
  | of `func`.
  |
  | Then the logical dimension they are reducing
  | over is dim 0 but the physical dim is dim 1
  | (the first non-batch dimension)
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/VmapTransforms.h]

// Most PyTorch operators take 4 or fewer inputs.
pub const K_VMAP_TRANSFORM_STATIC_INPUT_SIZE: i64 = 4;

pub type VmapPhysicalViewVec = SmallVector<VmapPhysicalView,kVmapTransformStaticInputSize>;

/**
  | Pytorch generally advertises good performance
  | for <= 5 dims. (see ATen/core/DimVector.h). We
  | add a few extra dims (~3) for vmap dimensions
  | to get 8. Adjust this number as necessary
  |
  */
pub const K_VMAP_STATIC_DIM_VEC_SIZE: i64 = 8;

pub type VmapDimVector = SmallVector<i64,kVmapStaticDimVecSize>;

/**
  | NOTE: [What is an VmapTransform?]
  |
  | An *VmapTransform* converts logical views of
  | tensors to physical views.
  |
  | Batching rules use VmapTransforms to convert
  | logical arguments to physical arguments, then
  | call one or more  operator that handles the
  | physical arguments, and then converts the
  | physical result back to a logical argument.
  |
  | VmapTransform for operators that take tensors
  | with multiple batch dims.
  |
  | Given one or more logical views on Tensors,
  | `logicalToPhysical` permutes all of the batch
  | dims to the front of the tensor, aligns and
  | expands the batch dims to match each other
  | (according to their `level`), and returns
  | a VmapPhysicalView on the tensor(s).
  |
  */
pub struct MultiBatchVmapTransform {

}

impl MultiBatchVmapTransform {

    pub fn logical_to_physical(logical_tensor: &Tensor) -> VmapPhysicalView {
        
        todo!();
        /*
        
        */
    }
    
    pub fn logical_to_physical(logical_tensors: &[Tensor]) -> VmapPhysicalViewVec {
        
        todo!();
        /*
        
        */
    }
}

/**
  | VmapTransform for operators that broadcast all
  | inputs.
  |
  | Given some logical views on Tensors,
  | `logicalToPhysical`:
  |
  | - permutes all of the batch dims to the front
  | of the tensors
  |
  | - aligns all the batch dims to the collective
  | levels of all of the tensors.
  |
  |   If a tensor does not have a batch dim for
  |   a vmap level, then it receives a size-one
  |   dimension for said level.
  |
  | - aligns the non-batch dims to have the same
  |   dimensionality, adding extra size-1
  |   dimensions in between the batch dimensions
  |   and the non-batch dimensions so that the
  |   batch dimensions are lined up from the right.
  |
  | For example: given inputs of size (B, 2) and
  | (B, 3, 2) where B is the batch dimension,
  | BroadcastingVmapTransform returns
  | VmapPhysicalViews that wrap tensors of size (B,
  | 1, 2) and (B, 3, 2).
  |
  | Given inputs of size (B, 2) and (2,),
  | BroadcastingVmapTransform returns
  | VmapPhysicalViews wrapping tensors of size (B,
  | 2) and (1, 2).
  |
  | We don't actually *need* to return a tensor of
  | size (1, 2) for the second tensor because the
  | broadcasting operation takes care of that for
  | us, but we do it anyways to keep things
  |
  */
pub struct BroadcastingVmapTransform {

}

impl BroadcastingVmapTransform {
    
    pub fn logical_to_physical(logical_tensors: &[Tensor]) -> VmapPhysicalViewVec {
        
        todo!();
        /*
        
        */
    }
}

/**
  | NOTE: [What is a VmapPhysicalView?]
  |
  | VmapPhysicalView represents a physical view on
  | a Tensor.
  |
  | One can use it to further convert logical
  | dimension indices, logical shapes, and more to
  | their physical variants, or convert a new
  | (physical) tensor into a logical
  | BatchedTensor. (TODO(rzou): some of these are
  | not yet implemented).
  |
  | VmapPhysicalView stores a physical tensor with
  | all of its batch dimensions at the front and
  | some levels that correspond to said batch
  | dimensions.
  |
  | The levels bitset specifies which vmap levels
  | correspond to the batch dimensions at the front
  | of the tensor. In particular, the number of set
  | bits corresponds to the number of batch
  | dimensions on `tensor` and the rightmost bit of
  | `levels` specifies the maximum number of nested
  | vmaps we are in at this point in time.
  |
  | For example, given:
  |
  |   physical_view = VmapPhysicalView(tensor=ones(2, 3, 4, 5, 6), levels={1, 3})
  |
  | Rightmost bit of `levels` is 3 indicating the
  | number of nested vmaps less than or equal to 3.
  |
  |   bitset: 010100
  |              ^
  |              |
  |   levels: 012345
  |
  */
pub struct VmapPhysicalView {
    levels: BitSet<kVmapNumLevels>,
    tensor: Tensor,
}

impl VmapPhysicalView {
    
    pub fn new<'a>(
        tensor: Tensor,
        levels: BitSet<kVmapNumLevels>) -> Self {
    
        todo!();
        /*
        : levels(levels),
        : tensor(tensor),

            TORCH_INTERNAL_ASSERT(!isBatchedTensor(tensor));
        */
    }
    
    pub fn tensor(&mut self) -> &'a mut Tensor {
        
        todo!();
        /*
            return tensor_;
        */
    }
    
    pub fn tensor(&self) -> &Tensor {
        
        todo!();
        /*
            return tensor_;
        */
    }

    /**
      | Maps logical dim indices to physical dim
      | indices. Also does dim wrapping.
      |
      | For example, given:
      |   physical_view = VmapPhysicalView(tensor=ones(2, 3, 4, 5), levels={1, 3})
      |
      | Then physical_view.getPhysicalDims({0, 1})
      | returns {2, 3}.
      |
      | This is because the size of levels tell us
      | that the first two dimensions of `tensor_`
      | are batch dimensions, so a logical dim of `n`
      | is actually a physical dim of `n + 2`.
      |
      */
    pub fn get_physical_dims(&self, logical_dims: &[i32]) -> VmapDimVector {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_physical_dim(&self, logical_dim: i64) -> i64 {
        
        todo!();
        /*
        
        */
    }

    /**
      | Returns a VmapPhysicalToLogicalMap object.
      |
      | This can be used for mapping a physical
      | tensor to a new logical tensor
      | (BatchedTensor)
      |
      */
    pub fn get_physical_to_logical_map(&self) -> VmapPhysicalToLogicalMap {
        
        todo!();
        /*
        
        */
    }

    /**
      | Maps a logical shape to a physical shape by
      | pre-pending the batch sizes to the logical
      | shape.
      |
      */
    pub fn get_physical_shape(&self, logical_shape: &[i32]) -> VmapDimVector {
        
        todo!();
        /*
        
        */
    }
    
    pub fn num_batch_dims(&self) -> i64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn num_logical_dims(&self) -> i64 {
        
        todo!();
        /*
        
        */
    }
}

/**
  | Convenience struct used for mapping a physical
  | tensor (a non-BatchedTensor) to a logical one
  | (BatchedTensor).
  |
  | It holds some levels that are used to do the
  | mapping and assumes that the batch dimensions
  | in the physical tensor all occur at the front
  | of the tensor.
  |
  */
pub struct VmapPhysicalToLogicalMap {
    levels: BitSet<kVmapNumLevels>,
}

impl VmapPhysicalToLogicalMap {
    
    pub fn new(levels: BitSet<kVmapNumLevels>) -> Self {
    
        todo!();
        /*
        : levels(levels),
        
        */
    }

    /**
      | Maps a physical tensor to a new logical
      | tensor (BatchedTensor).
      |
      | Assumes that all of the "batch dimensions"
      | are at the front of the physical tensor. For
      | example, given:
      |
      | - x = rank-4 Tensor with size 2, 3, 5, 7
      |
      | - levels = (2, 4)
      |
      | Returns:
      |
      | - BatchedTensor(x, bdims=[(dim=0,lvl=2),
      | (dim=1, lvl=4)])
      |
      */
    pub fn apply(&self, physical_tensor: &Tensor) -> Tensor {
        
        todo!();
        /*
        
        */
    }

    /**
      | Given a vector of physical tensors,
      |
      | 1. maps each tensor to a new logical
      |    tensor. Assumes that all of the "batch
      |    dimensions" are at the front of the
      |    physical tensors.
      |
      | 2. stores the new logical tensors back into
      |    the passed-in vector. This is to avoid
      |    additional dynamic allocations.
      |
      */
    pub fn apply_inplace(&self, physical_tensors: &mut Vec<Tensor>)  {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/VmapTransforms.cpp]

/**
  | Checks if the batch dims in `bdims` appear
  | at the front of the tensor.
  |
  */
pub fn are_bdims_at_front_in_order(bdims: BatchDimsRef) -> bool {
    
    todo!();
        /*
      for (i64 idx = 0; idx < bdims.size(); idx++) {
        if (bdims[idx].dim() != idx) {
          return false;
        }
      }
      return true;
        */
}

/**
  | Takes a BatchedTensorImpl, permutes
  | all of the batch dims to the front, and
  | then returns a physical version of the
  | Tensor.
  |
  */
pub fn permute_batch_dims_to_front(batched: *mut BatchedTensorImpl) -> Tensor {
    
    todo!();
        /*
            auto bdims = batched->bdims();
      const Tensor& physical_tensor = batched->value();
      if (areBdimsAtFrontInOrder(bdims)) {
        return physical_tensor;
      }
      const auto sizes = physical_tensor.sizes();
      VmapDimVector permutation(sizes.size(), 0);
      permutation.reserve(sizes.size());
      const auto is_bdim = createBatchDimBitset(bdims);
      i64 idx = 0;
      for (const auto& bdim : bdims) {
        permutation[idx++] = bdim.dim();
      }
      for (i64 ptr = 0; idx < sizes.size(); ptr++) {
        if (is_bdim[ptr]) {
          continue;
        }
        permutation[idx++] = ptr;
      }
      return physical_tensor.permute(permutation);
        */
}

impl MultiBatchVmapTransform {
    
    pub fn logical_to_physical(&mut self, logical_tensor: &Tensor) -> VmapPhysicalView {
        
        todo!();
        /*
            auto* batched = maybeGetBatchedImpl(logical_tensor);
      TORCH_INTERNAL_ASSERT(
          batched,
          "logicalToPhysical(tensor) should only be passed a BatchedTensor");
      return { permuteBatchDimsToFront(batched), createVmapLevelsBitset(batched->bdims()) };
        */
    }
}

impl VmapPhysicalView {
    
    pub fn num_batch_dims(&self) -> i64 {
        
        todo!();
        /*
            return levels_.count();
        */
    }
    
    pub fn num_logical_dims(&self) -> i64 {
        
        todo!();
        /*
            return /*physical*/tensor_.dim() - numBatchDims();
        */
    }
    
    pub fn get_physical_dims(&self, logical_dims: &[i32]) -> VmapDimVector {
        
        todo!();
        /*
            auto logical_ndim = numLogicalDims();
      // NB: fmap doesn't have a SmallVector variant, so we don't use it here.
      VmapDimVector result;
      result.reserve(logical_ndim);
      for (auto dim : logical_dims) {
        result.push_back(maybe_wrap_dim(dim, logical_ndim) + numBatchDims());
      }
      return result;
        */
    }
    
    pub fn get_physical_dim(&self, logical_dim: i64) -> i64 {
        
        todo!();
        /*
            auto logical_ndim = numLogicalDims();
      return maybe_wrap_dim(logical_dim, logical_ndim) + numBatchDims();
        */
    }
    
    pub fn get_physical_shape(&self, logical_shape: &[i32]) -> VmapDimVector {
        
        todo!();
        /*
            VmapDimVector result;
      result.reserve(logical_shape.size() + numBatchDims());
      auto tensor_sizes = tensor_.sizes();
      result.insert(result.end(), tensor_sizes.begin(), tensor_sizes.begin() + numBatchDims());
      result.insert(result.end(), logical_shape.begin(), logical_shape.end());
      return result;
        */
    }
}

pub fn compute_front_batch_dims_from_levels(levels_bitset: BitSet<kVmapNumLevels>) -> BatchDims {
    
    todo!();
        /*
            BatchDims bdims;
      i64 dim = 0;
      for (i64 level = 0; level < kVmapNumLevels; level++) {
        if (!levels_bitset[level]) {
          continue;
        }
        bdims.emplace_back(level, dim++);
      }
      return bdims;
        */
}

/**
  | Given a Tensor or a BatchedTensor, returns the
  | underlying physical tensor with all vmapped
  | dimensions permuted to the front, if they
  | exist, and a bitset of vmap levels that were
  | present in the tensor.
  |
  */
pub fn get_physical_tensor_and_levels(self_: &Tensor) -> (Tensor,BitSet<kVmapNumLevels>) {
    
    todo!();
        /*
            auto* batched = maybeGetBatchedImpl(self);
      if (batched) {
        return {permuteBatchDimsToFront(batched), createVmapLevelsBitset(batched->bdims())};
      }
      return {self, 0};
        */
}

/**
  | Given a Tensor or a BatchedTensor, creates
  | a physical view of the tensor such that it has
  | a batch dimension for each level in
  | `requested_levels` and `requested_example_dim`
  | number of non-batch-dimensions.
  |
  | This function is useful in preparing physical
  | views on tensors that can then be passed into
  | broadcasting operations.
  |
  | For example, when adding two BatchedTensors of
  | sizes [B0, 3] and [B0, B1, 2, 3], where the Bi
  | are the batch dimensions, we must align the
  | batch dimensions and non-batch-dimensions
  | (henceforth referred to as the "example"
  | dimensions) separately to produce tensors of
  | size [B0, 1, 1, 3] and [B0, B1, 2, 3] so that
  | they can be added.
  |
  | Here's a direct example of using
  | alignBatchDimsAtFront on the above two tensors.
  |
  | 1) alignBatchDimsAtFront([B0, 3],
  | requested_levels={0, 1},
  | requested_example_dim=2) returns a physical
  | view of size [B0, 1, 1, 3] by adding an extra
  | dimension for level 1 and another extra
  | dimension to pad the example dimensions to 2.
  |
  | 2) alignBatchDimsAtFront([B0, B1, 2, 3],
  | requested_levels={0, 1},
  | requested_example_dim=2) returns a physical
  | view of size [B0, B1, 2, 3]
  |
  */
pub fn align_batch_dims_at_front(
    self_:                 &Tensor,
    requested_levels:      BitSet<kVmapNumLevels>,
    requested_example_dim: i64) -> Tensor {
    
    todo!();
        /*
            Tensor physical_tensor;
      bitset<kVmapNumLevels> tensor_levels;
      tie(physical_tensor, tensor_levels) = getPhysicalTensorAndLevels(self);

      TORCH_INTERNAL_ASSERT(
        (tensor_levels | requested_levels) == requested_levels,
        "`requested_levels` must be a superset of `self`'s levels");

      auto physical_sizes = physical_tensor.sizes();

      auto tensor_example_dim = physical_sizes.size() - /*num_batch_dims*/tensor_levels.count();
      TORCH_INTERNAL_ASSERT(tensor_example_dim <= requested_example_dim);

      if (tensor_levels == requested_levels && tensor_example_dim == requested_example_dim) {
        // Optimization: no need to do another view if the physical tensor is
        // already the correct shape
        return physical_tensor;
      }

      VmapDimVector aligned_sizes(requested_levels.count() + requested_example_dim, 1);

      // align the example dims (non-bdims dims) first
      // aligned_sizes[-tensor_example_dim:] = tensor_sizes[-tensor_example_dim:]
      copy(
          physical_sizes.rbegin(),
          physical_sizes.rbegin() + tensor_example_dim,
          aligned_sizes.rbegin());

      // align the bdims
      i64 level = 0;
      i64 tensor_dim = 0;
      for (i64 bdim = 0; bdim < requested_levels.count(); bdim++) {
        // Determine the level of the bdim
        while (!requested_levels[level]) level++;
        if (tensor_levels[level]) {
          aligned_sizes[bdim] = physical_sizes[tensor_dim++];
        }
        level++;
      }
      return physical_tensor.view(aligned_sizes);
        */
}

impl MultiBatchVmapTransform {
    
    /**
      | The algorithm is as follows:
      |
      | 1. Figure out what all of the collective levels
      | in `logical_tensors` is.
      |
      | 2. Move all batch dims to the front of the
      |    tensors and add extra dims of size 1. At
      |    this point, every tensor will have
      |    a dimension for each of the collective
      |    levels.
      |
      | 3. Compute the batch_sizes.
      |
      | 4. Expand each physical tensor so that they
      |    have output batch size equal to
      |    `batch_sizes`
      |
      */
    pub fn logical_to_physical(&mut self, logical_tensors: &[Tensor]) -> VmapPhysicalViewVec {
        
        todo!();
        /*
            // Figure out all of the collective vmap levels in `logical_tensors`.
      bitset<kVmapNumLevels> collective_levels;
      for (const auto& logical_tensor : logical_tensors) {
        auto* batched = maybeGetBatchedImpl(logical_tensor);
        if (batched) {
          collective_levels |= createVmapLevelsBitset(batched->bdims());
        }
      }

      // Populate physical_tensors.
      // This contains a list of regular (non-Batched) Tensors where all of the
      // batch dims have been moved to the front of the tensor. Any previously
      // non-existing batch dims get added to the tensors as new dimensions of size 1.
      vector<Tensor> physical_tensors;
      i64 num_batch_dims = collective_levels.count();
      for (const auto& logical_tensor : logical_tensors) {
        auto requested_example_dim = /*logical_dim*/logical_tensor.dim();
        auto physical_tensor = alignBatchDimsAtFront(
            logical_tensor, collective_levels, requested_example_dim);
        physical_tensors.push_back(move(physical_tensor));
      }

      // Compute batch_sizes
      VmapDimVector batch_sizes(num_batch_dims, 1);
      for (const auto& physical_tensor : physical_tensors) {
        auto physical_sizes = physical_tensor.sizes();
        for (i64 dim = 0; dim < num_batch_dims; dim++) {
          if (physical_sizes[dim] != 1) {
            batch_sizes[dim] = physical_sizes[dim];
          }
        }
      }

      // Expand each physical_tensor so that it has batch sizes `batch_sizes`
      VmapPhysicalViewVec result;
      for (const auto& physical_tensor : physical_tensors) {
        VmapDimVector expanded_size(batch_sizes.begin(), batch_sizes.end());
        auto physical_sizes = physical_tensor.sizes();
        expanded_size.insert(
            expanded_size.end(),
            physical_sizes.begin() + num_batch_dims,
            physical_sizes.end());
        result.emplace_back(physical_tensor.expand(expanded_size), collective_levels);
      }
      return result;
        */
    }
}

pub fn get_levels_and_largest_logical_dim(logical_tensors: &[Tensor]) -> (BitSet<kVmapNumLevels>,i64) {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(logical_tensors.size() > 0);
      bitset<kVmapNumLevels> levels;
      i64 largest_logical_dim = -1;
      for (const auto& tensor : logical_tensors) {
        auto* batched = maybeGetBatchedImpl(tensor);
        if (batched) {
          levels = levels | createVmapLevelsBitset(batched->bdims());
        }
        auto tensor_logical_dim = /*logical dim*/tensor.dim();
        if (tensor_logical_dim > largest_logical_dim) {
          largest_logical_dim = tensor_logical_dim;
        }
      }
      return { levels, largest_logical_dim };
        */
}

impl BroadcastingVmapTransform {

    pub fn logical_to_physical(&mut self, logical_tensors: &[Tensor]) -> VmapPhysicalViewVec {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          logical_tensors.size() == 2,
          "This function has only been tested for two tensors. Please add more tests ",
          "before removing this check ");

      VmapPhysicalViewVec result;

      bitset<kVmapNumLevels> levels;
      i64 largest_logical_dim;
      tie(levels, largest_logical_dim) = getLevelsAndLargestLogicalDim(logical_tensors);

      for (const auto& tensor : logical_tensors) {
        // NB: It's possible that we didn't actually need to align `tensor`.
        // For example, when adding two tensors of size (B, 2), and (3, 2), where
        // the first Tensor is a BatchedTensor with batch dim B and the second is
        // a regular Tensor, we will return views of size (B, 1, 2) and (1, 3, 2).
        // However, the view on the second tensor is unnecessary: broadcasting
        // semantics allow for the addition of two tensors of size (B, 1, 2) and (3, 2)!
        //
        // If this unnecessary view is a problem, consider optimizing it away in
        // the future. This may involve creating a new type of VmapPhysicalView
        auto aligned = alignBatchDimsAtFront(tensor, levels, largest_logical_dim) ;
        result.emplace_back(move(aligned), levels);
      }
      return result;
        */
    }
}

impl VmapPhysicalView {
    
    pub fn get_physical_to_logical_map(&self) -> VmapPhysicalToLogicalMap {
        
        todo!();
        /*
            return VmapPhysicalToLogicalMap(levels_);
        */
    }
}

impl VmapPhysicalToLogicalMap {
    
    pub fn apply(&self, physical_tensor: &Tensor) -> Tensor {
        
        todo!();
        /*
            return makeBatched(physical_tensor, computeFrontBatchDimsFromLevels(levels_));
        */
    }
}

impl VmapPhysicalToLogicalMap {
    
    pub fn apply_inplace(&self, physical_tensors: &mut Vec<Tensor>)  {
        
        todo!();
        /*
      for (i64 idx = 0; idx < physical_tensors.size(); ++idx) {
        physical_tensors[idx] = apply(physical_tensors[idx]);
      }
        */
    }
}
