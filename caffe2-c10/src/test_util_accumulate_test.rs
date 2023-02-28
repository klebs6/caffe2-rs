crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/util/accumulate_test.cpp]

#[test] fn accumulate_test_vector() {
    todo!();
    /*
    
      vector<int> ints = {1, 2, 3, 4, 5};

      EXPECT_EQ(sum_integers(ints), 1 + 2 + 3 + 4 + 5);
      EXPECT_EQ(multiply_integers(ints), 1 * 2 * 3 * 4 * 5);

      EXPECT_EQ(sum_integers(ints.begin(), ints.end()), 1 + 2 + 3 + 4 + 5);
      EXPECT_EQ(
          multiply_integers(ints.begin(), ints.end()), 1 * 2 * 3 * 4 * 5);

      EXPECT_EQ(sum_integers(ints.begin() + 1, ints.end() - 1), 2 + 3 + 4);
      EXPECT_EQ(
          multiply_integers(ints.begin() + 1, ints.end() - 1), 2 * 3 * 4);

      EXPECT_EQ(numelements_from_dim(2, ints), 3 * 4 * 5);
      EXPECT_EQ(numelements_to_dim(3, ints), 1 * 2 * 3);
      EXPECT_EQ(numelements_between_dim(2, 4, ints), 3 * 4);
      EXPECT_EQ(numelements_between_dim(4, 2, ints), 3 * 4);

    */
}

#[test] fn accumulate_test_list() {
    todo!();
    /*
    
      list<int> ints = {1, 2, 3, 4, 5};

      EXPECT_EQ(sum_integers(ints), 1 + 2 + 3 + 4 + 5);
      EXPECT_EQ(multiply_integers(ints), 1 * 2 * 3 * 4 * 5);

      EXPECT_EQ(sum_integers(ints.begin(), ints.end()), 1 + 2 + 3 + 4 + 5);
      EXPECT_EQ(
          multiply_integers(ints.begin(), ints.end()), 1 * 2 * 3 * 4 * 5);

      EXPECT_EQ(numelements_from_dim(2, ints), 3 * 4 * 5);
      EXPECT_EQ(numelements_to_dim(3, ints), 1 * 2 * 3);
      EXPECT_EQ(numelements_between_dim(2, 4, ints), 3 * 4);
      EXPECT_EQ(numelements_between_dim(4, 2, ints), 3 * 4);

    */
}

#[test] fn accumulate_test_base_cases() {
    todo!();
    /*
    
      vector<int> ints = {};

      EXPECT_EQ(sum_integers(ints), 0);
      EXPECT_EQ(multiply_integers(ints), 1);

    */
}

#[test] fn accumulate_test_errors() {
    todo!();
    /*
    
      vector<int> ints = {1, 2, 3, 4, 5};

    #ifndef NDEBUG
      EXPECT_THROW(numelements_from_dim(-1, ints), Error);
    #endif

      EXPECT_THROW(numelements_to_dim(-1, ints), Error);
      EXPECT_THROW(numelements_between_dim(-1, 10, ints), Error);
      EXPECT_THROW(numelements_between_dim(10, -1, ints), Error);

      EXPECT_EQ(numelements_from_dim(10, ints), 1);
      EXPECT_THROW(numelements_to_dim(10, ints), Error);
      EXPECT_THROW(numelements_between_dim(10, 4, ints), Error);
      EXPECT_THROW(numelements_between_dim(4, 10, ints), Error);

    */
}
