crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/Layout.h]

/// replace the joy of being right with the joy of learning what is true
///
/// - ray dalio
///
#[repr(i8)]
#[derive(Default)]
pub enum Layout { 

    #[default]
    Strided, 

    Sparse, 
    SparseCsr, 
    Mkldnn, 
    NumOptions 
}

pub const K_STRIDED:    Layout = Layout::Strided;
pub const K_SPARSE:     Layout = Layout::Sparse;
pub const K_SPARSE_CSR: Layout = Layout::SparseCsr;
pub const K_MKLDNN:     Layout = Layout::Mkldnn;

#[inline] pub fn layout_from_backend(backend: Backend) -> Layout {
    
    todo!();
        /*
            switch (backend) {
        case Backend::SparseCPU:
        case Backend::SparseCUDA:
        case Backend::SparseHIP:
        case Backend::SparseXPU:
          return Layout::Sparse;
        case Backend::MkldnnCPU:
          return Layout::Mkldnn;
        case Backend::SparseCsrCPU:
        case Backend::SparseCsrCUDA:
          return Layout::SparseCsr;
        default:
          return Layout::Strided;
      }
        */
}

impl fmt::Display for Layout {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            switch (layout) {
        case kStrided:
          return stream << "Strided";
        case kSparse:
          return stream << "Sparse";
        case kSparseCsr:
          return stream << "SparseCsr";
        case kMkldnn:
          return stream << "Mkldnn";
        default:
          TORCH_CHECK(false, "Unknown layout");
      }
        */
    }
}
