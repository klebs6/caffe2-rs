crate::ix!();

#[test] fn AlgorithmsCacheTest_CachesCorrectly() {
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

#[test] fn AlgorithmsCacheTest_KeysDifferIfOneVectorIsEmpty() {
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

#[test] fn AlgorithmsCacheTest_KeysDifferIfFlagsAreDifferent() {
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
