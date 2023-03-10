crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/core/impl/SizesAndStrides_test.cpp]

pub fn check_data(
    sz:      &SizesAndStrides,
    sizes:   &[i32],
    strides: &[i32])  {
    
    todo!();
        /*
            EXPECT_EQ(sizes.size(), strides.size())
          << "bad test case: size() of sizes and strides don't match";
      EXPECT_EQ(sz.size(), sizes.size());

      int idx = 0;
      for (auto x : sizes) {
        EXPECT_EQ(sz.size_at_unchecked(idx), x) << "index: " << idx;
        EXPECT_EQ(sz.size_at(idx), x) << "index: " << idx;
        EXPECT_EQ(sz.sizes_data()[idx], x) << "index: " << idx;
        EXPECT_EQ(*(sz.sizes_begin() + idx), x) << "index: " << idx;
        idx++;
      }
      EXPECT_EQ(sz.sizes_arrayref(), sizes);

      idx = 0;
      for (auto x : strides) {
        EXPECT_EQ(sz.stride_at_unchecked(idx), x) << "index: " << idx;
        EXPECT_EQ(sz.stride_at(idx), x) << "index: " << idx;
        EXPECT_EQ(sz.strides_data()[idx], x) << "index: " << idx;
        EXPECT_EQ(*(sz.strides_begin() + idx), x) << "index: " << idx;

        idx++;
      }
      EXPECT_EQ(sz.strides_arrayref(), strides);
        */
}

#[test] fn sizes_and_strides_test_default_constructor() {
    todo!();
    /*
    
      SizesAndStrides sz;
      checkData(sz, {0}, {1});
      // Can't test size_at() out of bounds because it just asserts for now.

    */
}

#[test] fn sizes_and_strides_test_set() {
    todo!();
    /*
    
      SizesAndStrides sz;
      sz.set_sizes({5, 6, 7, 8});
      checkData(sz, {5, 6, 7, 8}, {1, 0, 0, 0});

    */
}

#[test] fn sizes_and_strides_test_resize() {
    todo!();
    /*
    
      SizesAndStrides sz;

      sz.resize(2);

      // Small to small growing.
      checkData(sz, {0, 0}, {1, 0});

      // Small to small growing, again.
      sz.resize(5);
      checkData(sz, {0, 0, 0, 0, 0}, {1, 0, 0, 0, 0});

      for (const auto ii : irange(sz.size())) {
        sz.size_at_unchecked(ii) = ii + 1;
        sz.stride_at_unchecked(ii) = 2 * (ii + 1);
      }

      checkData(sz, {1, 2, 3, 4, 5}, {2, 4, 6, 8, 10});

      // Small to small, shrinking.
      sz.resize(4);
      checkData(sz, {1, 2, 3, 4}, {2, 4, 6, 8});

      // Small to small with no size change.
      sz.resize(4);
      checkData(sz, {1, 2, 3, 4}, {2, 4, 6, 8});

      // Small to small, growing back so that we can confirm that our "new"
      // data really does get zeroed.
      sz.resize(5);
      checkData(sz, {1, 2, 3, 4, 0}, {2, 4, 6, 8, 0});

      // Small to big.
      sz.resize(6);

      checkData(sz, {1, 2, 3, 4, 0, 0}, {2, 4, 6, 8, 0, 0});

      sz.size_at_unchecked(5) = 6;
      sz.stride_at_unchecked(5) = 12;

      checkData(sz, {1, 2, 3, 4, 0, 6}, {2, 4, 6, 8, 0, 12});

      // Big to big, growing.
      sz.resize(7);

      checkData(sz, {1, 2, 3, 4, 0, 6, 0}, {2, 4, 6, 8, 0, 12, 0});

      // Big to big with no size change.
      sz.resize(7);

      checkData(sz, {1, 2, 3, 4, 0, 6, 0}, {2, 4, 6, 8, 0, 12, 0});

      sz.size_at_unchecked(6) = 11;
      sz.stride_at_unchecked(6) = 22;

      checkData(sz, {1, 2, 3, 4, 0, 6, 11}, {2, 4, 6, 8, 0, 12, 22});

      // Big to big, shrinking.
      sz.resize(6);
      checkData(sz, {1, 2, 3, 4, 0, 6}, {2, 4, 6, 8, 0, 12});

      // Grow back to make sure "new" elements get zeroed in big mode too.
      sz.resize(7);
      checkData(sz, {1, 2, 3, 4, 0, 6, 0}, {2, 4, 6, 8, 0, 12, 0});

      // Finally, big to small.

      // Give it different data than it had when it was small to avoid
      // getting it right by accident (i.e., because of leftover inline
      // storage when going small to big).
      for (const auto ii : irange(sz.size())) {
        sz.size_at_unchecked(ii) = ii - 1;
        sz.stride_at_unchecked(ii) = 2 * (ii - 1);
      }

      checkData(sz, {-1, 0, 1, 2, 3, 4, 5}, {-2, 0, 2, 4, 6, 8, 10});

      sz.resize(5);
      checkData(sz, {-1, 0, 1, 2, 3}, {-2, 0, 2, 4, 6});

    */
}

#[test] fn sizes_and_strides_test_set_at_index() {
    todo!();
    /*
    
      SizesAndStrides sz;

      sz.resize(5);
      sz.size_at(4) = 42;
      sz.stride_at(4) = 23;

      checkData(sz, {0, 0, 0, 0, 42}, {1, 0, 0, 0, 23});

      sz.resize(6);
      sz.size_at(5) = 43;
      sz.stride_at(5) = 24;

      checkData(sz, {0, 0, 0, 0, 42, 43}, {1, 0, 0, 0, 23, 24});

    */
}

#[test] fn sizes_and_strides_test_set_at_iterator() {
    todo!();
    /*
    
      SizesAndStrides sz;

      sz.resize(5);
      *(sz.sizes_begin() + 4) = 42;
      *(sz.strides_begin() + 4) = 23;

      checkData(sz, {0, 0, 0, 0, 42}, {1, 0, 0, 0, 23});

      sz.resize(6);
      *(sz.sizes_begin() + 5) = 43;
      *(sz.strides_begin() + 5) = 24;

      checkData(sz, {0, 0, 0, 0, 42, 43}, {1, 0, 0, 0, 23, 24});

    */
}

#[test] fn sizes_and_strides_test_set_via_data() {
    todo!();
    /*
    
      SizesAndStrides sz;

      sz.resize(5);
      *(sz.sizes_data() + 4) = 42;
      *(sz.strides_data() + 4) = 23;

      checkData(sz, {0, 0, 0, 0, 42}, {1, 0, 0, 0, 23});

      sz.resize(6);
      *(sz.sizes_data() + 5) = 43;
      *(sz.strides_data() + 5) = 24;

      checkData(sz, {0, 0, 0, 0, 42, 43}, {1, 0, 0, 0, 23, 24});

    */
}

pub fn make_small(offset: Option<i32>) -> SizesAndStrides {

    let offset: i32 = offset.unwrap_or(0);

    todo!();
        /*
            SizesAndStrides small;
      small.resize(3);
      for (const auto ii : irange(small.size())) {
        small.size_at_unchecked(ii) = ii + 1 + offset;
        small.stride_at_unchecked(ii) = 2 * (ii + 1 + offset);
      }

      return small;
        */
}

pub fn make_big(offset: Option<i32>) -> SizesAndStrides {

    let offset: i32 = offset.unwrap_or(0);

    todo!();
        /*
            SizesAndStrides big;
      big.resize(8);
      for (const auto ii : irange(big.size())) {
        big.size_at_unchecked(ii) = ii - 1 + offset;
        big.stride_at_unchecked(ii) = 2 * (ii - 1 + offset);
      }

      return big;
        */
}

pub fn check_small(
        sm:     &SizesAndStrides,
        offset: Option<i32>) {

    let offset: i32 = offset.unwrap_or(0);

    todo!();
        /*
            vector<int64_t> sizes(3), strides(3);
      for (int ii = 0; ii < 3; ++ii) {
        sizes[ii] = ii + 1 + offset;
        strides[ii] = 2 * (ii + 1 + offset);
      }
      checkData(sm, sizes, strides);
        */
}

pub fn check_big(
    big:    &SizesAndStrides,
    offset: Option<i32>)  {

    let offset: i32 = offset.unwrap_or(0);

    todo!();
        /*
            vector<int64_t> sizes(8), strides(8);
      for (int ii = 0; ii < 8; ++ii) {
        sizes[ii] = ii - 1 + offset;
        strides[ii] = 2 * (ii - 1 + offset);
      }
      checkData(big, sizes, strides);
        */
}

#[test] fn sizes_and_strides_test_move_constructor() {
    todo!();
    /*
    
      SizesAndStrides empty;

      SizesAndStrides movedEmpty(move(empty));

      EXPECT_EQ(empty.size(), 0);
      EXPECT_EQ(movedEmpty.size(), 1);
      checkData(movedEmpty, {0}, {1});

      SizesAndStrides small = makeSmall();
      checkSmall(small);

      SizesAndStrides movedSmall(move(small));
      checkSmall(movedSmall);
      EXPECT_EQ(small.size(), 0);

      SizesAndStrides big = makeBig();
      checkBig(big);

      SizesAndStrides movedBig(move(big));
      checkBig(movedBig);
      EXPECT_EQ(big.size(), 0);

    */
}

#[test] fn sizes_and_strides_test_copy_constructor() {
    todo!();
    /*
    
      SizesAndStrides empty;

      SizesAndStrides copiedEmpty(empty);

      EXPECT_EQ(empty.size(), 1);
      EXPECT_EQ(copiedEmpty.size(), 1);
      checkData(empty, {0}, {1});
      checkData(copiedEmpty, {0}, {1});

      SizesAndStrides small = makeSmall();
      checkSmall(small);

      SizesAndStrides copiedSmall(small);
      checkSmall(copiedSmall);
      checkSmall(small);

      SizesAndStrides big = makeBig();
      checkBig(big);

      SizesAndStrides copiedBig(big);
      checkBig(big);
      checkBig(copiedBig);

    */
}

#[test] fn sizes_and_strides_test_copy_assignment_small_to() {
    todo!();
    /*
    
      SizesAndStrides smallTarget = makeSmall();
      SizesAndStrides smallCopyFrom = makeSmall(1);

      checkSmall(smallTarget);
      checkSmall(smallCopyFrom, 1);

      smallTarget = smallCopyFrom;

      checkSmall(smallTarget, 1);
      checkSmall(smallCopyFrom, 1);

    */
}

#[test] fn sizes_and_strides_test_move_assignment_small_to() {
    todo!();
    /*
    
      SizesAndStrides smallTarget = makeSmall();
      SizesAndStrides smallMoveFrom = makeSmall(1);

      checkSmall(smallTarget);
      checkSmall(smallMoveFrom, 1);

      smallTarget = move(smallMoveFrom);

      checkSmall(smallTarget, 1);
      EXPECT_EQ(smallMoveFrom.size(), 0);

    */
}

#[test] fn sizes_and_strides_test_copy_assignment_small_to_big() {
    todo!();
    /*
    
      SizesAndStrides bigTarget = makeBig();
      SizesAndStrides smallCopyFrom = makeSmall();

      checkBig(bigTarget);
      checkSmall(smallCopyFrom);

      bigTarget = smallCopyFrom;

      checkSmall(bigTarget);
      checkSmall(smallCopyFrom);

    */
}

#[test] fn sizes_and_strides_test_move_assignment_small_to_big() {
    todo!();
    /*
    
      SizesAndStrides bigTarget = makeBig();
      SizesAndStrides smallMoveFrom = makeSmall();

      checkBig(bigTarget);
      checkSmall(smallMoveFrom);

      bigTarget = move(smallMoveFrom);

      checkSmall(bigTarget);
      EXPECT_EQ(smallMoveFrom.size(), 0);

    */
}

#[test] fn sizes_and_strides_test_copy_assignment_big_to() {
    todo!();
    /*
    
      SizesAndStrides bigTarget = makeBig();
      SizesAndStrides bigCopyFrom = makeBig(1);

      checkBig(bigTarget);
      checkBig(bigCopyFrom, 1);

      bigTarget = bigCopyFrom;

      checkBig(bigTarget, 1);
      checkBig(bigCopyFrom, 1);

    */
}

#[test] fn sizes_and_strides_test_move_assignment_big_to() {
    todo!();
    /*
    
      SizesAndStrides bigTarget = makeBig();
      SizesAndStrides bigMoveFrom = makeBig(1);

      checkBig(bigTarget);
      checkBig(bigMoveFrom, 1);

      bigTarget = move(bigMoveFrom);

      checkBig(bigTarget, 1);
      EXPECT_EQ(bigMoveFrom.size(), 0);

    */
}

#[test] fn sizes_and_strides_test_copy_assignment_big_to_small() {
    todo!();
    /*
    
      SizesAndStrides smallTarget = makeSmall();
      SizesAndStrides bigCopyFrom = makeBig();

      checkSmall(smallTarget);
      checkBig(bigCopyFrom);

      smallTarget = bigCopyFrom;

      checkBig(smallTarget);
      checkBig(bigCopyFrom);

    */
}

#[test] fn sizes_and_strides_test_move_assignment_big_to_small() {
    todo!();
    /*
    
      SizesAndStrides smallTarget = makeSmall();
      SizesAndStrides bigMoveFrom = makeBig();

      checkSmall(smallTarget);
      checkBig(bigMoveFrom);

      smallTarget = move(bigMoveFrom);

      checkBig(smallTarget);
      EXPECT_EQ(bigMoveFrom.size(), 0);

    */
}

#[test] fn sizes_and_strides_test_copy_assignment_self() {
    todo!();
    /*
    
      SizesAndStrides small = makeSmall();
      SizesAndStrides big = makeBig();

      checkSmall(small);
      checkBig(big);

      small = small;
      checkSmall(small);

      big = big;
      checkBig(big);

    */
}

// Avoid failures due to -Wall -Wself-move.
pub fn self_move(
        x: &mut SizesAndStrides,
        y: &mut SizesAndStrides)  {
    
    todo!();
        /*
            x = move(y);
        */
}

#[test] fn sizes_and_strides_test_move_assignment_self() {
    todo!();
    /*
    
      SizesAndStrides small = makeSmall();
      SizesAndStrides big = makeBig();

      checkSmall(small);
      checkBig(big);

      selfMove(small, small);
      checkSmall(small);

      selfMove(big, big);
      checkBig(big);

    */
}
