crate::ix!();

#[inline] pub fn get_topK<T>(
    input:            *const T,
    n:                i64,
    k:                i64,
    src_offset:       i64,
    dst_offset:       i64,
    stride:           i64,
    values:           *mut T,
    indices:          *mut i64,
    flatten_indices:  *mut i64) 
{
    todo!();
    /*
        const T* src_ptr = input + src_offset;
      std::vector<std::pair<T, int64_t>> heap_data;
      heap_data.reserve(k);
      for (int64_t i = 0; i < k && i < n; ++i) {
        heap_data.emplace_back(*src_ptr, i);
        src_ptr += stride;
      }
      std::priority_queue<
          std::pair<T, int64_t>,
          std::vector<std::pair<T, int64_t>>,
          ValueComp<T>>
          pq(ValueComp<T>(), std::move(heap_data));
      for (int64_t i = k; i < n; ++i) {
        if (pq.top().first < *src_ptr) {
          pq.pop();
          pq.emplace(*src_ptr, i);
        }
        src_ptr += stride;
      }
      int64_t dst_pos = dst_offset + (std::min(k, n) - 1) * stride;
      while (!pq.empty()) {
        const auto& item = pq.top();
        values[dst_pos] = item.first;
        indices[dst_pos] = item.second;
        if (flatten_indices != nullptr) {
          flatten_indices[dst_pos] = src_offset + item.second * stride;
        }
        pq.pop();
        dst_pos -= stride;
      }
    */
}
