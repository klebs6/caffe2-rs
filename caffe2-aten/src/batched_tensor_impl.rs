crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/BatchedTensorImpl.h]

/**
  | We assume this in a few other places in
  | the codebase, but there isn't a centralized
  | definition.
  |
  */
pub const VMAP_MAX_TENSOR_DIMS: i64 = 64;

/**
  | The valid vmap levels range from [0,
  | 64). This effectively means that we
  | support a maximum of 64 nested vmaps.
  |
  */
pub const VMAP_NUM_LEVELS: i64 = 64;

/**
  | Store this number of elements of BatchDims on
  | the stack. Most people will probably use <=
  | 5 nested vmaps, but adjust this number as
  | necessary.
  */
pub const BATCH_DIMS_STACK_SIZE: i64 = 5;

/**
  | a BatchDim represents a "private" dimension on
  | a Tensor created inside of vmap. It is
  | a (level, dim) tuple, with the `dim` indicating
  | which dimension is being vmap'ed over and the
  | `level` being an identifier for which vmap said
  | dimension was created inside.
  |
  | The `dim` corresponds to a "physical dim" - it
  | is a dimension index on the underlying physical
  | tensor that is being vmapped over.
  */
pub struct BatchDim {
    dim:   i64,
    level: i64,
}

impl BatchDim {
    
    pub fn new(
        level: i64,
        dim:   i64) -> Self {
    
        todo!();
        /*
        : dim(dim),
        : level(level),
        */
    }
    
    pub fn dim(&self) -> i64 {
        
        todo!();
        /*
            return dim_;
        */
    }
    
    pub fn level(&self) -> i64 {
        
        todo!();
        /*
            return level_;
        */
    }
}

pub type BatchDims        = SmallVector<BatchDim,kBatchDimsStackSize>;
pub type BatchDimsRef<'a> = &'a [BatchDim];

/**
  | A BatchedTensorImpl holds an underlying Tensor
  | and a list of BatchDim
  |
  | NB: We use the term "BatchedTensor" to mean
  | a Tensor that is backed with
  | a BatchedTensorImpl.
  |
  | The batch dimensions are treated as being
  | "private"; they are not user-visible.
  |
  | For example, in the following Tensor,
  |    bt = BatchedTensorImpl(
  |    ones(2, 3, 5, 7),
  |    [(lvl=1, dim=0), (lvl=2, dim=1)]
  |    )
  |
  | dimensions 0 and 1 are batch dimensions.
  |
  | bt.sizes() returns (5, 7); bt.sum(0) performs
  | a reduction over the (public) dim 0, which is
  | equivalent to dim 3 in the underlying ones(2,
  | 3, 5, 7) tensor.
  */
pub struct BatchedTensorImpl {
    base: TensorImpl,

    value: Tensor,

    /**
      | -----------
      | @note
      | 
      | [BatchedTensorImpl levels invariant]
      | 
      | There is an invariant that the BatchDims
      | must be stored in increasing `level`
      | order.
      | 
      | That is, for i < j, bdims_[i].level must
      | be less than bdims_[j].level.
      |
      */
    bdims: BatchDims,
}

impl BatchedTensorImpl {
    
    pub fn new(
        value: Tensor,
        bdims: BatchDims) -> Self {
    
        todo!();
        /*


        
        */
    }

    /**
      | Returns a reference to BatchDims that
      | represent which dimensions of this tensor are
      | private.
      |
      */
    pub fn bdims<'a>(&'a self) -> BatchDimsRef<'a> {
        
        todo!();
        /*
            return bdims_;
        */
    }

    /**
      | BatchedTensorImpl wraps a Tensor
      |
      */
    pub fn value(&self) -> &Tensor {
        
        todo!();
        /*
            return value_; }{
        */
    }

    /**
      | Given a public dimension index, return the
      | dimension index in the underlying value()
      | tensor.
      |
      | For example, if we have
      |    bt = BatchedTensorImpl(ones(2, 3, 5, 7), [(lvl=1, dim=0), (lvl=2, dim=2)])
      | bt.actualDim(0) -> 1
      | bt.actualDim(1) -> 3
      | bt.actualDim(2) -> Error
      */
    pub fn actual_dim(&self, 
        dim:      i64,
        wrap_dim: bool) -> i64 {
        let wrap_dim: bool = wrap_dim.unwrap_or(true);

        todo!();
        /*
        
        */
    }

    /**
      | Override a bunch of methods inherited
      | from
      | 
      | TensorImpl to return error messages.
      |
      */
    pub fn is_contiguous_custom(&self, memory_format: MemoryFormat) -> bool {
        
        todo!();
        /*
        
        */
    }

    pub fn set_size(&mut self, 
        dim:      i64,
        new_size: i64)  {
        
        todo!();
        /*
        
        */
    }

    pub fn set_stride(&mut self, 
        dim:        i64,
        new_stride: i64)  {
        
        todo!();
        /*
        
        */
    }

    pub fn set_storage_offset(&mut self, storage_offset: i64)  {
        
        todo!();
        /*
        
        */
    }

    #[cfg(debug_assertions)]
    pub fn has_storage(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
 
    /**
      | see NOTE: [BatchedTensorImpl levels
      | invariant]
      |
      */
    pub fn check_invariants(&self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn tensorimpl_type_name(&self) -> *const u8 {
        
        todo!();
        /*
        
        */
    }
}

/**
  | NB: We use the term "BatchedTensor" to mean
  | a Tensor that is backed with
  | a BatchedTensorImpl.
  |
  */
#[inline] pub fn is_batched_tensor(tensor: &Tensor) -> bool {
    
    todo!();
        /*
            return tensor.unsafeGetTensorImpl()->key_set().has(DispatchKey::Batched);
        */
}

/**
  | It is unsafe to call this on a Tensor that is
  | not backed by a BatchedTensorImpl. Please use
  | `maybeGetBatchedImpl` whenever possible.
  |
  */
#[inline] pub fn unsafe_get_batched_impl(tensor: Tensor) -> *mut BatchedTensorImpl {
    
    todo!();
        /*
            return static_cast<BatchedTensorImpl*>(tensor.unsafeGetTensorImpl());
        */
}

#[inline] pub fn maybe_get_batched_impl(tensor: Tensor) -> *mut BatchedTensorImpl {
    
    todo!();
        /*
            if (!isBatchedTensor(tensor)) {
        return nullptr;
      }
      return unsafeGetBatchedImpl(tensor);
        */
}

/**
  | Returns a bitset. If bit i is set, then
  | that means dim i is a batchdim.
  |
  */
#[inline] pub fn create_batch_dim_bitset<'a>(bdims: BatchDimsRef<'a>) -> BitSet<kVmapMaxTensorDims> {
    
    todo!();
        /*
            bitset<kVmapMaxTensorDims> is_bdim;
      for (const auto& bdim : bdims) {
        is_bdim.set(bdim.dim());
      }
      return is_bdim;
        */
}

/**
  | Creates a bitset for all of the levels
  | present in `bdims`
  |
  */
#[inline] pub fn create_vmap_levels_bitset<'a>(bdims: BatchDimsRef<'a>) -> BitSet<kVmapNumLevels> {
    
    todo!();
        /*
            bitset<kVmapNumLevels> result;
      for (const auto& bdim : bdims) {
        result.set(bdim.level());
      }
      return result;
        */
}

impl fmt::Display for &mut std::io::BufWriter {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            out << "(lvl=" << bdim.level() << ", dim=" << bdim.dim() << ")";
      return out;
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/BatchedTensorImpl.cpp]

impl BatchedTensorImpl {
    
    pub fn new(
        value: Tensor,
        bdims: BatchDims) -> Self {
    
        todo!();
        /*
        : tensor_impl(DispatchKeySet(DispatchKey::Batched),
              value.dtype(),
              value.device()
            ),
        : value(move(value)),
        : bdims(move(bdims)),

            TORCH_INTERNAL_ASSERT(value_.defined());
      set_storage_access_should_throw();
      set_has_contiguity_policy(HasContiguityPolicy::CustomBehavior);
      checkInvariants();

      const auto public_dims = value_.dim() - bdims_.size();
      const auto value_sizes = value_.sizes();
      const auto value_strides = value_.strides();
      sizes_and_strides_.resize(public_dims);
      for (const auto dim : irange(public_dims)) {
        auto actual_dim = actualDim(dim, /*wrap_dim=*/false);
        sizes_and_strides_.size_at_unchecked(dim) = value_sizes.at(actual_dim);
        sizes_and_strides_.stride_at_unchecked(dim) = value_strides.at(actual_dim);
      }
      refresh_numel();
      refresh_contiguous();
        */
    }
   
    pub fn actual_dim(&self, 
        dim:      i64,
        wrap_dim: bool) -> i64 {
        
        todo!();
        /*
            if (wrap_dim) {
        const auto ndim = sizes_and_strides_.size();
        dim = maybe_wrap_dim(dim, ndim);
      }
      auto is_bdim = createBatchDimBitset(bdims_);

      // Example: assume dim = 3, and is_bdim = 10010011000...
      // The 1's are batch dims and 0's are normal dims of the underlying value_ Tensor.
      // actualDim gives us the index of `dim` in the `value_` Tensor, which is equivalent
      // to asking "where does the 3rd (0-indexed) zero occur in the bitset?".
      // The answer to that is index 5.
      //
      // TODO(rzou): the PDEP instruction does exactly this
      // (https://stackoverflow.com/questions/7669057/find-nth-set-bit-in-an-int)
      // but it might require newer (>= ~2015) CPUs. We should clean this up
      // if/when we have dropped support for older CPUs.
      i64 non_bdim_count = 0;
      for (const auto actual_dim : irange(kVmapMaxTensorDims)) {
        if (is_bdim[actual_dim]) {
          continue;
        }
        if (non_bdim_count == dim) {
          return actual_dim;
        }
        non_bdim_count++;
      }
      // If we hit this assert, then that means
      // `non_bdim_count` + #num_bdims > kVmapMaxTensorDims. We restrict the number
      // of dims a BatchedTensorImpl can have to kVmapMaxTensorDims so this should
      // never be hit.
      TORCH_INTERNAL_ASSERT(false);
        */
    }
    
    pub fn check_invariants(&self)  {
        
        todo!();
        /*
            i64 prev_level = -1;
      for (const auto& bdim : bdims_) {
        TORCH_INTERNAL_ASSERT(bdim.level() > prev_level);
        prev_level = bdim.level();
      }
        */
    }

    /**
      | The following are publically exposed
      | as methods of Tensor
      |
      */
    pub fn is_contiguous_custom(&self, memory_format: MemoryFormat) -> bool {
        
        todo!();
        /*
            TORCH_CHECK(memory_format == MemoryFormat::Contiguous,
          "NYI: querying is_contiguous inside of vmap for memory_format ",
          "other than torch.contiguous_format");
      return is_contiguous_;
        */
    }
    
    /**
      | The following are some internal inherited
      | methods that we do not support.
      |
      | They should never get called.
      */
    pub fn set_size(&mut self, 
        dim:      i64,
        new_size: i64)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "Can't set_size for BatchedTensorImpl");
        */
    }
    
    pub fn set_stride(&mut self, 
        dim:        i64,
        new_stride: i64)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "Can't set_stride for BatchedTensorImpl");
        */
    }
    
    pub fn set_storage_offset(&mut self, storage_offset: i64)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "Can't set_storage_offset for BatchedTensorImpl");
        */
    }
    
    pub fn tensorimpl_type_name(&self) -> *const u8 {
        
        todo!();
        /*
            return "BatchedTensorImpl";
        */
    }
    
    #[cfg(debug_assertions)]
    pub fn has_storage(&self) -> bool {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!storage_, "BatchedTensorImpl assumes that storage_ is never set");
      return false;
        */
    }
}

/**
  | Use this to construct a BatchedTensor
  | from a regular Tensor
  |
  */
pub fn make_batched(
        tensor: &Tensor,
        bdims:  BatchDims) -> Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(!isBatchedTensor(tensor));
      auto tensor_dim = tensor.dim();
      TORCH_CHECK(
          tensor_dim <= kVmapMaxTensorDims,
          "vmap only supports tensors of dimensionality up to ", kVmapMaxTensorDims,
          "; got a tensor with dim ", tensor_dim);
      TORCH_INTERNAL_ASSERT(
          all_of(bdims.begin(), bdims.end(),
              [](const BatchDim& bdim) { return bdim.level() < kVmapNumLevels; }),
          "We only support up to ", kVmapNumLevels, " nested vmaps");
      return detail::make_tensor<BatchedTensorImpl>(tensor, move(bdims));
        */
}

/**
  | Adds a batch dim to `tensor`, returning
  | a BatchedTensor
  |
  */
pub fn add_batch_dim(
        tensor: &Tensor,
        level:  i64,
        dim:    i64) -> Tensor {
    
    todo!();
        /*
            const auto* batched = maybeGetBatchedImpl(tensor);
      if (!batched) {
        BatchDims bdims;
        bdims.emplace_back(level, dim);
        return detail::make_tensor<BatchedTensorImpl>(tensor, move(bdims));
      }
      BatchDims new_bdims(batched->bdims().begin(), batched->bdims().end());
      auto actual_bdim = batched->actualDim(dim, /*wrap_dim=*/true);
      new_bdims.emplace_back(level, actual_bdim);
      return makeBatched(batched->value(), move(new_bdims));
        */
}

/**
  | Checks if an inplace operation on self and
  | other is "vmap compatible".
  |
  | See NOTE: [vmap-incompatible in-place
  | operations] for the definition of this.
  |
  */
pub fn inplace_is_vmap_compatible(
        self_: &Tensor,
        other: &Tensor) -> bool {
    
    todo!();
        /*
            const auto* other_batched = maybeGetBatchedImpl(other);
      if (!other_batched) {
        return true;
      }
      const auto* self_batched = maybeGetBatchedImpl(self);
      if (!self_batched) {
        // self is not batched but other is batched
        return false;
      }
      auto self_levels = createVmapLevelsBitset(self_batched->bdims());
      auto other_levels = createVmapLevelsBitset(other_batched->bdims());
      return self_levels == (self_levels | other_levels);
        */
}

