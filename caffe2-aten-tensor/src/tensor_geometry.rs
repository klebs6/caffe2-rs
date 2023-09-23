crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorGeometry.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorGeometry.cpp]

pub struct TensorGeometry {
    sizes:          Vec<i64>,
    strides:        Vec<i64>,
    storage_offset: i64,
    numel:          i64,
}

impl Default for TensorGeometry {
    
    fn default() -> Self {
        todo!();
        /*
        : storage_offset(0),

        
        */
    }
}

impl TensorGeometry {
    
    pub fn new(sizes: &[i32]) -> Self {
    
        todo!();
        /*


            : sizes_(sizes.vec())
        , strides_(sizes.size())
        , storage_offset_(0) 
          i64 dim = sizes.size();
          i64 expected_stride = 1;
          for (i64 i = dim - 1; i >= 0; i--) {
            strides_[i] = expected_stride;
            expected_stride *= sizes_[i];
          }
          numel_ = expected_stride;
        */
    }
    
    pub fn new(t: &Tensor) -> Self {
    
        todo!();
        /*


            : sizes_(t.sizes().vec())
        , strides_(t.strides().vec())
        , storage_offset_(t.storage_offset())
        , numel_(t.numel())
        */
    }

    /// true if the tensor is contiguous
    ///
    pub fn is_contiguous(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn dim(&self) -> i64 {
        
        todo!();
        /*
            return sizes_.size();
        */
    }
    
    pub fn size(&self, dim: i64) -> i64 {
        
        todo!();
        /*
            dim = maybe_wrap_dim(dim, this->dim());
        return sizes_.at(static_cast<usize>(dim));
        */
    }
    
    pub fn sizes(&self) -> &[i32] {
        
        todo!();
        /*
            return IntArrayRef{ sizes_ };
        */
    }
    
    pub fn stride(&self, dim: i64) -> i64 {
        
        todo!();
        /*
            dim = maybe_wrap_dim(dim, this->dim());
        return strides_.at(static_cast<usize>(dim));
        */
    }
    
    pub fn strides(&self) -> &[i32] {
        
        todo!();
        /*
            return IntArrayRef{ strides_ };
        */
    }
    
    pub fn storage_offset(&self) -> i64 {
        
        todo!();
        /*
            return storage_offset_;
        */
    }
    
    pub fn numel(&self) -> i64 {
        
        todo!();
        /*
            return numel_;
        */
    }
    
    pub fn transpose(&mut self, 
        dim0: i64,
        dim1: i64) -> TensorGeometry {
        
        todo!();
        /*
            TensorGeometry r = *this; // copy
        TORCH_CHECK(dim0 < dim(), "transpose: dim0=", dim0, " out of range (dim=", dim(), ")")
        TORCH_CHECK(dim1 < dim(), "transpose: dim1=", dim1, " out of range (dim=", dim(), ")")
        swap(r.sizes_[dim0], r.sizes_[dim1]);
        swap(r.strides_[dim0], r.strides_[dim1]);
        return r;
        */
    }
    
    pub fn is_contiguous(&self) -> bool {
        
        todo!();
        /*
            if (numel_ == 0) {
        return true;
      }
      return geometry_is_contiguous(sizes_, strides_);
        */
    }
}
