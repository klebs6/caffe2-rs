crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/impl/SizesAndStrides.h]

pub const C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE: usize = 5;

/**
  | Packed container for TensorImpl sizes and
  | strides.
  |
  | This design improves on the previous approach
  | of using a pair of SmallVector<int64_t, 5> by
  | specializing for the operations we actually use
  | and enforcing that the number of sizes is the
  | same as the number of strides.
  |
  | The memory layout is as follows:
  |
  | 1 size_t for the size
  |
  | 5 eightbytes of inline sizes and 5 eightbytes
  | of inline strides, OR pointer to out-of-line
  | array
  |
  */
pub union SizesAndStridesUnion {
    out_of_line_storage: *mut i64,
    inline_storage:      [i64; C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE * 2],
}

pub struct SizesAndStrides {
    size: usize,
    u:    SizesAndStridesUnion,
}

/**
  | TODO: different iterator types for sizes
  | & strides to prevent mixing the two
  | accidentally.
  |
  */
pub type SizesIterator        = *mut i64;
pub type SizesConstIterator   = *const i64;
pub type StridesIterator      = *mut i64;
pub type StridesConstIterator = *const i64;

impl Default for SizesAndStrides {
    
    fn default() -> Self {
        todo!();
        /*
        : size(1),

            size_at_unchecked(0) = 0;
        stride_at_unchecked(0) = 1;
        */
    }
}

impl Drop for SizesAndStrides {

    fn drop(&mut self) {
        todo!();
        /*
            if (C10_UNLIKELY(!isInline())) {
          free(outOfLineStorage_);
        }
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/core/impl/SizesAndStrides.cpp]
impl SizesAndStrides {

    
    pub fn new_from_ref(rhs: &SizesAndStrides) -> Self {
    
        todo!();
        /*
        : size(rhs.size_),

            if (C10_LIKELY(rhs.isInline())) {
          copyDataInline(rhs);
        } else {
          allocateOutOfLineStorage(size_);
          copyDataOutline(rhs);
        }
        */
    }
    
    pub fn assign_from_ref(&mut self, rhs: &SizesAndStrides) -> &mut SizesAndStrides {
        
        todo!();
        /*
            if (this == &rhs) {
          return *this;
        }
        if (C10_LIKELY(rhs.isInline())) {
          if (C10_UNLIKELY(!isInline())) {
            free(outOfLineStorage_);
          }
          copyDataInline(rhs);
        } else {
          if (isInline()) {
            allocateOutOfLineStorage(rhs.size_);
          } else {
            resizeOutOfLineStorage(rhs.size_);
          }
          copyDataOutline(rhs);
        }
        size_ = rhs.size_;
        return *this;
        */
    }

    /// Move from rhs. rhs.size() == 0 afterwards.
    ///
    pub fn new(rhs: SizesAndStrides) -> Self {
    
        todo!();
        /*
        : size(rhs.size_),

            if (C10_LIKELY(isInline())) {
          memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
        } else {
          outOfLineStorage_ = rhs.outOfLineStorage_;
          rhs.outOfLineStorage_ = nullptr;
        }

        rhs.size_ = 0;
        */
    }

    /// Move from rhs. rhs.size() == 0 afterwards.
    ///
    pub fn assign_from(&mut self, rhs: SizesAndStrides) -> &mut SizesAndStrides {
        
        todo!();
        /*
            if (this == &rhs) {
          return *this;
        }
        if (C10_LIKELY(rhs.isInline())) {
          if (C10_UNLIKELY(!isInline())) {
            free(outOfLineStorage_);
          }
          copyDataInline(rhs);
        } else {
          // They're outline. We're going to steal their vector.
          if (!isInline()) {
            free(outOfLineStorage_);
          }
          outOfLineStorage_ = rhs.outOfLineStorage_;
          rhs.outOfLineStorage_ = nullptr;
        }
        size_ = rhs.size_;
        rhs.size_ = 0;

        return *this;
        */
    }
    
    pub fn size(&self) -> usize {
        
        todo!();
        /*
            return size_;
        */
    }
    
    pub fn sizes_data(&self) -> *const i64 {
        
        todo!();
        /*
            if (C10_LIKELY(isInline())) {
          return &inlineStorage_[0];
        } else {
          return &outOfLineStorage_[0];
        }
        */
    }
    
    pub fn sizes_data_mut(&mut self) -> *mut i64 {
        
        todo!();
        /*
            if (C10_LIKELY(isInline())) {
          return &inlineStorage_[0];
        } else {
          return &outOfLineStorage_[0];
        }
        */
    }
    
    
    pub fn sizes_begin(&self) -> SizesConstIterator {
        
        todo!();
        /*
            return sizes_data();
        */
    }
    
    pub fn sizes_begin_mut(&mut self) -> SizesIterator {
        
        todo!();
        /*
            return sizes_data();
        */
    }
    
    
    pub fn sizes_end(&self) -> SizesConstIterator {
        
        todo!();
        /*
            return sizes_begin() + size();
        */
    }
    
    pub fn sizes_end_mut(&mut self) -> SizesIterator {
        
        todo!();
        /*
            return sizes_begin() + size();
        */
    }
    
    pub fn sizes_arrayref(&self) -> &[i32] {
        
        todo!();
        /*
            return IntArrayRef{sizes_data(), size()};
        */
    }
    
    pub fn set_sizes(&mut self, new_sizes: &[i32])  {
        
        todo!();
        /*
            resize(newSizes.size());
        copy(newSizes.begin(), newSizes.end(), sizes_begin());
        */
    }
    
    pub fn strides_data(&self) -> *const i64 {
        
        todo!();
        /*
            if (C10_LIKELY(isInline())) {
          return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
        } else {
          return &outOfLineStorage_[size()];
        }
        */
    }
    
    pub fn strides_data_mut(&mut self) -> *mut i64 {
        
        todo!();
        /*
            if (C10_LIKELY(isInline())) {
          return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
        } else {
          return &outOfLineStorage_[size()];
        }
        */
    }
    
    pub fn strides_begin(&self) -> StridesConstIterator {
        
        todo!();
        /*
            if (C10_LIKELY(isInline())) {
          return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
        } else {
          return &outOfLineStorage_[size()];
        }
        */
    }
    
    pub fn strides_begin_mut(&mut self) -> StridesIterator {
        
        todo!();
        /*
            if (C10_LIKELY(isInline())) {
          return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
        } else {
          return &outOfLineStorage_[size()];
        }
        */
    }
    
    pub fn strides_end(&self) -> StridesConstIterator {
        
        todo!();
        /*
            return strides_begin() + size();
        */
    }
    
    pub fn strides_end_mut(&mut self) -> StridesIterator {
        
        todo!();
        /*
            return strides_begin() + size();
        */
    }
    
    pub fn strides_arrayref(&self) -> &[i32] {
        
        todo!();
        /*
            return IntArrayRef{strides_data(), size()};
        */
    }

    /// Size accessors.
    pub fn size_at(&self, idx: usize) -> i64 {
        
        todo!();
        /*
            assert(idx < size());
        return sizes_data()[idx];
        */
    }
    
    pub fn size_at_mut(&mut self, idx: usize) -> &mut i64 {
        
        todo!();
        /*
            assert(idx < size());
        return sizes_data()[idx];
        */
    }
    
    pub fn size_at_unchecked(&self, idx: usize) -> i64 {
        
        todo!();
        /*
            return sizes_data()[idx];
        */
    }
    
    pub fn size_at_unchecked_mut(&mut self, idx: usize) -> &mut i64 {
        
        todo!();
        /*
            return sizes_data()[idx];
        */
    }

    /// Size accessors.
    pub fn stride_at(&self, idx: usize) -> i64 {
        
        todo!();
        /*
            assert(idx < size());
        return strides_data()[idx];
        */
    }
    
    pub fn stride_at_mut(&mut self, idx: usize) -> &mut i64 {
        
        todo!();
        /*
            assert(idx < size());
        return strides_data()[idx];
        */
    }
    
    pub fn stride_at_unchecked(&self, idx: usize) -> i64 {
        
        todo!();
        /*
            return strides_data()[idx];
        */
    }
    
    pub fn stride_at_unchecked_mut(&mut self, idx: usize) -> &mut i64 {
        
        todo!();
        /*
            return strides_data()[idx];
        */
    }
    
    pub fn resize(&mut self, new_size: usize)  {
        
        todo!();
        /*
            const auto oldSize = size();
        if (newSize == oldSize) {
          return;
        }
        if (C10_LIKELY(
                newSize <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE && isInline())) {
          if (oldSize < newSize) {
            const auto bytesToZero =
                (newSize - oldSize) * sizeof(inlineStorage_[0]);
            memset(&inlineStorage_[oldSize], 0, bytesToZero);
            memset(
                &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE + oldSize],
                0,
                bytesToZero);
          }
          size_ = newSize;
        } else {
          resizeSlowPath(newSize, oldSize);
        }
        */
    }
    
    pub fn is_inline(&self) -> bool {
        
        todo!();
        /*
            return size_ <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE;
        */
    }
    
    
    pub fn copy_data_inline(&mut self, rhs: &SizesAndStrides)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.isInline());
        memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
        */
    }
    
    
    pub fn storage_bytes(size: usize) -> usize {
        
        todo!();
        /*
            return size * 2 * sizeof(int64_t);
        */
    }
    
    
    pub fn allocate_out_of_line_storage(&mut self, size: usize)  {
        
        todo!();
        /*
            outOfLineStorage_ = static_cast<int64_t*>(malloc(storageBytes(size)));
        TORCH_CHECK(
            outOfLineStorage_,
            "Could not allocate memory for Tensor SizesAndStrides!");
        */
    }
    
    
    pub fn resize_out_of_line_storage(&mut self, new_size: usize)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!isInline());
        outOfLineStorage_ = static_cast<int64_t*>(
            realloc(outOfLineStorage_, storageBytes(newSize)));
        TORCH_CHECK(
            outOfLineStorage_,
            "Could not allocate memory for Tensor SizesAndStrides!");
        */
    }
    
    
    pub fn copy_data_outline(&mut self, rhs: &SizesAndStrides)  {
        
        todo!();
        /*
            memcpy(outOfLineStorage_, rhs.outOfLineStorage_, storageBytes(rhs.size_));
        */
    }
    
    pub fn resize_slow_path(&mut self, 
        new_size: usize,
        old_size: usize)  {
        
        todo!();
        /*
            if (newSize <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE) {
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            !isInline(),
            "resizeSlowPath called when fast path should have been hit!");
        int64_t* tempStorage = outOfLineStorage_;
        memcpy(
            &inlineStorage_[0],
            &tempStorage[0],
            C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE * sizeof(inlineStorage_[0]));
        memcpy(
            &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE],
            &tempStorage[oldSize],
            C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE * sizeof(inlineStorage_[0]));
        // CANNOT USE freeOutOfLineStorage() HERE! outOfLineStorage_
        // HAS BEEN OVERWRITTEN!
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(tempStorage);
      } else {
        if (isInline()) {
          // CANNOT USE allocateOutOfLineStorage(newSize) HERE! WOULD
          // OVERWRITE inlineStorage_!
          // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
          int64_t* tempStorage =
              static_cast<int64_t*>(malloc(storageBytes(newSize)));
          TORCH_CHECK(
              tempStorage,
              "Could not allocate memory to change Tensor SizesAndStrides!");
          const auto bytesToCopy = oldSize * sizeof(inlineStorage_[0]);
          const auto bytesToZero = (newSize > oldSize)
              ? (newSize - oldSize) * sizeof(tempStorage[0])
              : 0;
          memcpy(&tempStorage[0], &inlineStorage_[0], bytesToCopy);
          if (bytesToZero) {
            memset(&tempStorage[oldSize], 0, bytesToZero);
          }
          memcpy(
              &tempStorage[newSize],
              &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE],
              bytesToCopy);
          if (bytesToZero) {
            memset(&tempStorage[newSize + oldSize], 0, bytesToZero);
          }
          outOfLineStorage_ = tempStorage;
        } else {
          const bool isGrowing = oldSize < newSize;
          if (isGrowing) {
            // Resize before shifting so that we have room.
            resizeOutOfLineStorage(newSize);
          }
          // Shift the old strides to their new starting point. Note
          // that this does not occur in the inline path above because
          // the stride starting point is not moving.
          memmove(
              outOfLineStorage_ + newSize,
              outOfLineStorage_ + oldSize,
              min(oldSize, newSize) * sizeof(outOfLineStorage_[0]));
          if (!isGrowing) {
            // Resize after shifting so that we don't lose data.
            resizeOutOfLineStorage(newSize);
          } else {
            // Zero the end of the sizes portion.
            const auto bytesToZero =
                (newSize - oldSize) * sizeof(outOfLineStorage_[0]);
            memset(&outOfLineStorage_[oldSize], 0, bytesToZero);
            memset(&outOfLineStorage_[newSize + oldSize], 0, bytesToZero);
          }
        }
      }
      size_ = newSize;
        */
    }
}
