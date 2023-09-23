crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/SparseCsrTensorUtils.h]

pub type SparseCsrTensor = Tensor;

#[inline] pub fn get_sparse_csr_impl(self_: &SparseCsrTensor) -> *mut SparseCsrTensorImpl {
    
    todo!();
        /*
            AT_ASSERTM(
          self.is_sparse_csr(),
          "_internal_get_SparseCsrTensorImpl: not a sparse CSR tensor");
      return static_cast<SparseCsrTensorImpl*>(self.unsafeGetTensorImpl());
        */
}
