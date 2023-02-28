crate::ix!();

/**
  | If there's an existing packed matrix
  | for the same matrix, reuse it.
  | 
  | Create a new one otherwise. This can
  | save memory usage if many threads are
  | sharing the same weight. i16, i32
  |
  */
#[cfg(use_fbgemm)]
#[inline] pub fn get_or_create_fbgemm_packb_matrix<ACC_T>(
    trans:          fbgemm::matrix_op_t,
    m:              i32,
    n:              i32,
    orig_data:      *const c_void,
    quantized_data: *const i8,
    ld:             i32) -> Arc<PackBMatrix<i8,ACC_T>> {

    todo!();
    /*
        static map<
          tuple<int, int, const void*>,
          weak_ptr<fbgemm::PackBMatrix<int8_t, ACC_T>>>
          cache;
      static mutex cache_mutex;

      // Create a new packed matrix and compare with cached one if there's any.
      // Note that a cache miss is as expensive as a cache hit here, the purpose of
      // this cache is only to deduplicate the quantized tensors for improved
      // memory bandwidth if different nets share copies of the same operator.
      // TODO: make this cheaper by computing hash of fdata.
      auto new_packed = make_shared<fbgemm::PackBMatrix<int8_t, ACC_T>>(
          trans,
          m,
          n,
          quantized_data,
          ld,
          nullptr, // pmat
          1); // groups

      tuple<int, int, const void*> key(m, n, orig_data);
      shared_ptr<fbgemm::PackBMatrix<int8_t, ACC_T>> cache_entry;
      {
        lock_guard<mutex> lock(cache_mutex);
        auto itr = cache.find(key);
        if (itr != cache.end()) {
          cache_entry = itr->second.lock();
        }
      } // release lock here during expensive equals()

      if (!cache_entry || !cache_entry->metaEquals(*new_packed) ||
          !cache_entry->equals(*new_packed)) {
        // cache miss
        lock_guard<mutex> lock(cache_mutex);
        cache[key] = new_packed;
        return new_packed;
      } else {
        return cache_entry;
      }
    */
}
