crate::ix!();

impl OneHotOp<CPUContext> {

    #[inline] pub fn do_one_hot_op(&mut self, 
        batch_size: i64,
        index_size: i64,
        indices:    &Tensor,
        one_hots:   *mut Tensor)  {
        
        todo!();
        /*
            const int64_t* indices_ptr = indices.template data<int64_t>();
      float* one_hots_ptr = one_hots->template mutable_data<float>();
      memset(one_hots_ptr, 0, one_hots->nbytes());
      for (int i = 0; i < batch_size; ++i) {
        auto label_idx = indices_ptr[i];
        DCHECK((0 <= label_idx) && (label_idx < index_size));
        one_hots_ptr[label_idx] = 1.0;
        one_hots_ptr += index_size;
      }
        */
    }
}
