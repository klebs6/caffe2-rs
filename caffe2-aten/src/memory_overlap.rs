crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/MemoryOverlap.h]

/**
  | MemOverlap: Whether or not there is memory
  | overlap
  |
  | NO: Absolutely no memory overlap
  |
  | YES: Absolutely yes memory overlap
  |
  | TOO_HARD: There might be memory overlap, but it
  | was too expensive to compute.
  |
  | NB: Please update the python test for these if
  | you renumber them.
  */
pub enum MemOverlap { 
    NO, 
    YES, 
    TOO_HARD 
}

pub enum MemOverlapStatus { 
    FULL, 
    PARTIAL, 
    NO, 
    TOO_HARD 
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/MemoryOverlap.cpp]

pub fn has_internal_overlap(tensor: &Tensor) -> MemOverlap {

    /*
    fn has_internal_overlap(t: *mut TensorImpl) -> MemOverlap {
        
        todo!();
            /*
                TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t->layout() == kStrided);

          if (t->is_non_overlapping_and_dense()) {
            return MemOverlap::NO;
          }

          auto strides = t->strides();
          auto sizes = t->sizes();
          for (usize i = 0; i < strides.size(); ++i) {
            if (strides[i] == 0 && sizes[i] > 1) {
              return MemOverlap::YES;
            }
          }

          return MemOverlap::TOO_HARD;
            */
    }
    */

    todo!();
        /*
            return has_internal_overlap(tensor.unsafeGetTensorImpl());
        */
}

pub fn assert_no_internal_overlap(t: &Tensor)  {

    /*
    pub fn assert_no_internal_overlap(t: *mut TensorImpl)  {
        
        todo!();
            /*
                TORCH_CHECK(has_internal_overlap(t) != MemOverlap::YES,
            "unsupported operation: more than one element of the written-to tensor "
            "refers to a single memory location. Please clone() the tensor before "
            "performing the operation.");
            */
    }
    */
    
    todo!();
        /*
            assert_no_internal_overlap(t.unsafeGetTensorImpl());
        */
}

pub fn get_overlap_status(
        a: &Tensor,
        b: &Tensor) -> MemOverlapStatus {

    /*
    pub fn get_overlap_status(
            a: *mut TensorImpl,
            b: *mut TensorImpl) -> MemOverlapStatus {
        
        todo!();
            /*
                if (a == b) return MemOverlapStatus::FULL;
          if (a->numel() == 0 || b->numel() == 0) {
            return MemOverlapStatus::NO;
          }
          if (!a->is_non_overlapping_and_dense() || !b->is_non_overlapping_and_dense()) {
            return MemOverlapStatus::TOO_HARD;
          }
          // Test for storage equality, rather than pointer equality.
          // This reduces precision, but if people are aliasing the
          // same pointer across multiple storages there are many
          // similar situations (e.g., storage().data() == storage().data()+1)
          // which we will miss.
          auto a_storage = a->unsafe_storage();
          if (a_storage && a_storage.is_alias_of(b->unsafe_storage())) {
            const auto a_begin = static_cast<char*>(a->data());
            const auto a_end = a_begin + a->numel() * a->itemsize();
            const auto b_begin = static_cast<char*>(b->data());
            const auto b_end = b_begin + b->numel() * b->itemsize();

            if (a_begin == b_begin && a_end == b_end) {
              return MemOverlapStatus::FULL;
            }
            if (a_begin < b_end && b_begin < a_end) {
              return MemOverlapStatus::PARTIAL;
            }
          }
          return MemOverlapStatus::NO;
            */
    }
    */
    
    todo!();
        /*
            return get_overlap_status(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
        */
}


pub fn assert_no_partial_overlap(
        a: &Tensor,
        b: &Tensor)  {

    /*
    pub fn assert_no_partial_overlap(
            a: *mut TensorImpl,
            b: *mut TensorImpl)  {
        
        todo!();
            /*
                TORCH_CHECK(get_overlap_status(a, b) != MemOverlapStatus::PARTIAL,
            "unsupported operation: some elements of the input tensor and "
            "the written-to tensor refer to a single memory location. "
            "Please clone() the tensor before performing the operation.");
            */
    }
    */
        
    todo!();
        /*
            assert_no_partial_overlap(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
        */
}

pub fn assert_no_overlap(
        a: &Tensor,
        b: &Tensor)  {

    /*
    pub fn assert_no_overlap(
            a: *mut TensorImpl,
            b: *mut TensorImpl)  {
        
        todo!();
            /*
                const auto lap = get_overlap_status(a, b);
          TORCH_CHECK(lap != MemOverlapStatus::PARTIAL && lap != MemOverlapStatus::FULL,
            "unsupported operation: some elements of the input tensor and "
            "the written-to tensor refer to a single memory location. "
            "Please clone() the tensor before performing the operation.");
            */
    }
    */
    
    todo!();
        /*
            assert_no_overlap(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
        */
}
