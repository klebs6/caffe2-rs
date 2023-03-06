crate::ix!();

/**
  | A sub tensor view
  | 
  | TODO: remove?
  |
  */
pub struct ConstTensorView<T> {
    data: *mut T,
    dims: Vec<i32>,
}

impl<T> ConstTensorView<T> {
    
    pub fn new(data: *const T, dims: &Vec<i32>) -> Self {
        todo!();
        /*
            : data_(data), dims_(dims)
        */
    }
    
    #[inline] pub fn ndim(&self) -> i32 {
        
        todo!();
        /*
            return dims_.size();
        */
    }
    
    #[inline] pub fn dims(&self) -> &Vec<i32> {
        
        todo!();
        /*
            return dims_;
        */
    }
    
    #[inline] pub fn dim(&self, i: i32) -> i32 {
        
        todo!();
        /*
            DCHECK_LE(i, dims_.size());
        return dims_[i];
        */
    }
    
    #[inline] pub fn data(&self) -> *const T {
        
        todo!();
        /*
            return data_;
        */
    }
    
    #[inline] pub fn size(&self) -> usize {
        
        todo!();
        /*
            return std::accumulate(
            dims_.begin(), dims_.end(), 1, std::multiplies<size_t>());
        */
    }
}
