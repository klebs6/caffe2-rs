crate::ix!();

//impl for:
//cudnnConvolutionFwdAlgo_t, 
//cudnnConvolutionBwdFilterAlgo_t, 
//cudnnConvolutionBwdDataAlgo_t, 
//int // for testing
pub struct AlgorithmsCache<TAlgorithm> {
    hash: HashMap<i64, TAlgorithm>,
}

impl<TAlgorithm> AlgorithmsCache<TAlgorithm> {

    /**
      | Caches the best algorithm for a given
      | combination of tensor dimensions & compute
      | data type.
      |
      */
    pub fn get_algorithm(
        &mut self,
        tensor_dim1:     &[i32],
        tensor_dim2:     &[i32],

        // Differentiate between algorithms with different parameters in a generic way
        algorithm_flags: i32,

        generating_func: fn() -> TAlgorithm) -> TAlgorithm 
    {
        todo!();
        /*
      int64_t seed = 0;
      // Hash all of the inputs, which we wiill then use to try and look up
      // a previously discovered algorithm, or fall back to generating a new one.
      std::hash<int64_t> hashFn;
      for (const auto num : tensorDimensions1) {
        // Copied from boost::hash_combine.
        // Adding 1 to differentiate between first and second vector.
        seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2) + 1;
      }

      for (const auto num : tensorDimensions2) {
        // Copied from boost::hash_combine.
        seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }

      // Adding 2 to differentiate from previous vectors
      seed ^= hashFn(algorithmFlags) + 0x9e3779b9 + (seed << 6) + (seed >> 2) + 2;

      if (seed == 0) {
        return generatingFunc();
      }

      if (hash_.find(seed) == hash_.end()) {
        TAlgorithm value = generatingFunc();
        hash_[seed] = value;
      }

      return hash_[seed];
        */
    }
}
