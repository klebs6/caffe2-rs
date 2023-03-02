crate::ix!();

/// used for lengths tensors in the dataset
pub type TLength = i32;

/**
  | used for all internal dataset operations
  | (offsets, sizes to read, etc.)
  |
  */
pub type TOffset = i64;

///--------------------------------------------------

pub struct TreeIteratorFieldDesc {
    id:              i32,
    length_field_id: i32, // = -1;
    name:            String,
}

///---------------------------------------
pub struct TreeCursor {
    offsets: Vec<TOffset>,
    mutex:   parking_lot::RawMutex,
    it:      TreeIterator,
}

impl TreeCursor {
    
    pub fn new(iterator: &TreeIterator) -> Self {
        todo!();
        /*
            : it(iterator)
        */
    }
}


pub type SharedTensorVectorPtr   = Arc<Vec<TensorCPU>>;
pub type Shared2DTensorVectorPtr = Arc<Vec<Vec<TensorCPU>>>;
pub type Tensor2DVector          = Vec<Vec<TensorCPU>>;
pub type TensorVectorPtr         = Box<Vec<Tensor>>;

