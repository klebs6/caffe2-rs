crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/MatrixRef.h]

pub type MatrixRefSizeType = usize;

/**
  | MatrixRef - Like an ArrayRef, but with an
  | extra recorded strides so that we can easily
  | view it as a multidimensional array.
  |
  | Like ArrayRef, this class does not own the
  | underlying data, it is expected to be used in
  | situations where the data resides in some
  | other buffer.
  |
  | This is intended to be trivially copyable, so
  | it should be passed by value.
  |
  | For now, 2D only (so the copies are actually
  | cheap, without having to write a SmallVector
  | class) and contiguous only (so we can return
  | non-strided ArrayRef on index).
  |
  | P.S. dimension 0 indexes rows, dimension
  | 1 indexes columns
  */
pub struct MatrixRef<'a,T> {

    /**
      | Underlying ArrayRef
      |
      */
    arr:     &'a [T],

    /**
      | Stride of dim 0 (outer dimension)
      |
      */
    stride0: SizeType,

    // Stride of dim 1 is assumed to be 1
}

impl<'a,T> Default for MatrixRef<'a,T> {
    
    /// Construct an empty Matrixref.
    ///
    fn default() -> Self {
        todo!();
        /*


            : arr(nullptr), stride0(0)
        */
    }
}

impl<'a,T> Index<usize> for MatrixRef<'a,T> {

    type Output = &[T];
    
    #[inline] fn index(&self, index: usize) -> &Self::Output {
        todo!();
        /*
            return arr.slice(Index*stride0, stride0);
        */
    }
}

impl<'a,T> MatrixRef<'a,T> {

    /// Construct an MatrixRef from an ArrayRef
    /// and outer stride.
    ///
    pub fn new(
        arr:     &[T],
        stride0: SizeType) -> Self {
    
        todo!();
        /*


            : arr(arr), stride0(stride0) 

        TORCH_CHECK(arr.size() % stride0 == 0, "MatrixRef: ArrayRef size ", arr.size(), " not divisible by stride ", stride0)
        */
    }

    /// empty - Check if the matrix is empty.
    ///
    pub fn empty(&self) -> bool {
        
        todo!();
        /*
            return arr.empty();
        */
    }
    
    pub fn data(&self) -> *const T {
        
        todo!();
        /*
            return arr.data();
        */
    }

    /// size - Get size a dimension
    ///
    pub fn size(&self, dim: usize) -> usize {
        
        todo!();
        /*
            if (dim == 0) {
        return arr.size() / stride0;
      } else if (dim == 1) {
        return stride0;
      } else {
        TORCH_CHECK(0, "MatrixRef: out of bounds dimension ", dim, "; expected 0 or 1");
      }
        */
    }
    
    pub fn numel(&self) -> usize {
        
        todo!();
        /*
            return arr.size();
        */
    }

    /// equals - Check for element-wise equality.
    ///
    pub fn equals(&self, RHS: MatrixRef) -> bool {
        
        todo!();
        /*
            return stride0 == RHS.stride0 && arr.equals(RHS.arr);
        */
    }
}
