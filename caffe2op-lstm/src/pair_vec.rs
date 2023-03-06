crate::ix!();

/**
  | Gathers every two elements of a vector
  | in a vector of pairs
  |
  */
#[inline] pub fn pair_vec<T>(vals: &Vec<T>) -> Vec<(T,T)> {
    todo!();
    /*
        CAFFE_ENFORCE_EQ(
          vals.size() % 2,
          0,
          "Odd number of params or hiddens given to a bidirectional RNN");
      std::vector<std::pair<T, T>> result;
      result.reserve(vals.size() / 2);
      for (int64_t i = 0; i < vals.size(); i += 2) {
        result.emplace_back(copy_ctor(vals[i]), copy_ctor(vals[i + 1]));
      }
      return result;
    */
}

/// Flattens a vector of pairs
#[inline] pub fn unpair_vec<T>(vals: Vec<(T,T)>) -> Vec<T> {
    todo!();
    /*
        std::vector<T> result;
      result.reserve(vals.size() * 2);
      for (int64_t i = 0; i < vals.size(); i++) {
        result.push_back(std::move(vals[i].first));
        result.push_back(std::move(vals[i].second));
      }
      return result;
    */
}
