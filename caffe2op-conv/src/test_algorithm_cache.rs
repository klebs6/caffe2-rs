crate::ix!();

#[test] fn algorithms_cache_test_caches_correctly() {
    todo!();
    /*
  AlgorithmsCache<int> cache;
  int result = cache.getAlgorithm(
      std::vector<int64_t>(1), std::vector<int64_t>(1), 0, []() { return 5; });
  EXPECT_EQ(result, 5);

  int res2 = cache.getAlgorithm(
      std::vector<int64_t>(1), std::vector<int64_t>(1), 0, []() { return 10; });

  EXPECT_EQ(res2, 5);
  */
}

#[test] fn algorithms_cache_test_keys_differ_if_one_vector_is_empty() {
    todo!();
    /*
  AlgorithmsCache<int> cache;
  int result = cache.getAlgorithm(
      std::vector<int64_t>(1, 10), std::vector<int64_t>(), 0, []() { return 5; });
  EXPECT_EQ(result, 5);

  int res2 = cache.getAlgorithm(
      std::vector<int64_t>(), std::vector<int64_t>(1, 10), 0, []() {
        return 10;
      });

  EXPECT_EQ(res2, 10);
  */
}

#[test] fn algorithms_cache_test_keys_differ_if_flags_are_different() {
    todo!();
    /*
  AlgorithmsCache<int> cache;
  int result = cache.getAlgorithm(
      std::vector<int64_t>{2, 3, 4}, std::vector<int64_t>{5, 6}, 123, []() {
        return 5;
      });
  EXPECT_EQ(result, 5);

  int res2 = cache.getAlgorithm(
      std::vector<int64_t>{2, 3, 4}, std::vector<int64_t>{5, 6}, 456, []() {
        return 10;
      });

  EXPECT_EQ(res2, 10);

  int res3 = cache.getAlgorithm(
      std::vector<int64_t>{2, 3, 4}, std::vector<int64_t>{5, 6}, 456, []() {
        return 15;
      });

  EXPECT_EQ(res3, 10);
  */
}
