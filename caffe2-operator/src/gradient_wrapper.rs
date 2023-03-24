crate::ix!();

/**
  | -----------
  | @brief
  | 
  | A struct that abstracts on top of dense
  | and sparse blobs.
  | 
  | For a dense blob, its gradient name should
  | be written into dense_, and for a sparse
  | blob, its gradient name should be written
  | into indice_ for the sparse indices
  | and value_ for the values.
  |
  */
pub struct GradientWrapper {
    dense:   String,
    indices: String,
    values:  String,
}

impl GradientWrapper {

    #[inline] pub fn is_dense() -> bool {
        
        todo!();
        /*
            return (dense_.size() != 0);
        */
    }


    #[inline] pub fn is_sparse() -> bool {
        
        todo!();
        /*
            return (indices_.size() != 0 || values_.size() != 0);
        */
    }


    #[inline] pub fn is_empty() -> bool {
        
        todo!();
        /*
            return (!IsDense() && !IsSparse());
        */
    }
}

