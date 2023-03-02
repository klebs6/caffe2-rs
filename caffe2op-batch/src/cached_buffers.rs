crate::ix!();

/**
  | Buffers used by the MKL version are cached
  | across calls.
  |
  */
pub struct CachedBuffers {
    ty: i32,
}

struct TypedCachedBuffers<T> {
    ty:          i32,
    lambda1:     Vec<T>,
    lambda2:     Vec<T>,
    lambda2_z:   Vec<T>,
    accumulator: Vec<T>
}

// Helpers to access cached buffers.
#[macro_export] macro_rules! define_cached_buffers {
    ($t:ty, $tag:expr) => {
        /*
        template <>                                                                 
            template <>                                                                 
            BatchBoxCoxOp<CPUContext>::TypedCachedBuffers<T>&                           
            BatchBoxCoxOp<CPUContext>::GetBuffers<T>() {                                
                if (!buffers_ || buffers_->type_ != tag) {                                
                    buffers_.reset(new BatchBoxCoxOp<CPUContext>::TypedCachedBuffers<T>()); 
                    buffers_->type_ = tag;                                                  
                }                                                                         
                return *static_cast<TypedCachedBuffers<T>*>(buffers_.get());              
            }
        */
    }
}

define_cached_buffers!{f32, 1}
define_cached_buffers!{f64, 2}

