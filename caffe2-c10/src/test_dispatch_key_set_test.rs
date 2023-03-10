crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/core/DispatchKeySet_test.cpp]

#[test] fn dispatch_key_set_empty() {
    todo!();
    /*
    
      DispatchKeySet empty_set;
      for (uint8_t i = 1; i < static_cast<uint8_t>(DispatchKey::NumDispatchKeys);
           i++) {
        auto tid = static_cast<DispatchKey>(i);
        ASSERT_FALSE(empty_set.has(tid));
      }
      ASSERT_TRUE(empty_set.empty());
      DispatchKeySet empty_set2;
      ASSERT_TRUE(empty_set == empty_set2);
      ASSERT_EQ(empty_set.highestPriorityTypeId(), DispatchKey::Undefined);

    */
}

#[test] fn dispatch_key_set_singleton() {
    todo!();
    /*
    
      for (uint8_t i = 1; i < static_cast<uint8_t>(DispatchKey::NumDispatchKeys);
           i++) {
        auto tid = static_cast<DispatchKey>(i);
        DispatchKeySet sing(tid);
        ASSERT_EQ(sing, sing);
        ASSERT_EQ(sing, DispatchKeySet().add(tid));
        ASSERT_EQ(sing, sing.add(tid));
        ASSERT_EQ(sing, sing | sing);
        ASSERT_FALSE(sing.empty());
        ASSERT_TRUE(sing.has(tid));
        ASSERT_EQ(sing.highestPriorityTypeId(), tid);
        ASSERT_EQ(sing.remove(tid), DispatchKeySet());
      }

    */
}

#[test] fn dispatch_key_set_doubleton() {
    todo!();
    /*
    
      for (uint8_t i = 1; i < static_cast<uint8_t>(DispatchKey::NumDispatchKeys);
           i++) {
        for (uint8_t j = i + 1;
             j < static_cast<uint8_t>(DispatchKey::NumDispatchKeys);
             j++) {
          ASSERT_LT(i, j);
          auto tid1 = static_cast<DispatchKey>(i);
          auto tid2 = static_cast<DispatchKey>(j);
          auto doub = DispatchKeySet(tid1).add(tid2);
          ASSERT_EQ(doub, DispatchKeySet(tid1) | DispatchKeySet(tid2));
          ASSERT_TRUE(doub.has(tid1));
          ASSERT_TRUE(doub.has(tid2));
          ASSERT_EQ(doub.highestPriorityTypeId(), tid2); // relies on i < j
        }
      }

    */
}

#[test] fn dispatch_key_set_full() {
    todo!();
    /*
    
      DispatchKeySet full(DispatchKeySet::FULL);
      for (uint8_t i = 1; i < static_cast<uint8_t>(DispatchKey::NumDispatchKeys);
           i++) {
        auto tid = static_cast<DispatchKey>(i);
        ASSERT_TRUE(full.has(tid));
      }

    */
}

#[test] fn dispatch_key_set_iterator_basic_ops() {
    todo!();
    /*
    
      DispatchKeySet empty_set;
      DispatchKeySet full_set(DispatchKeySet::FULL);
      DispatchKeySet mutated_set = empty_set.add(static_cast<DispatchKey>(1));

      // Constructor + Comparison
      ASSERT_EQ(*empty_set.begin(), DispatchKey::NumDispatchKeys);
      ASSERT_EQ(*empty_set.end(), DispatchKey::NumDispatchKeys);
      ASSERT_EQ(*mutated_set.begin(), static_cast<DispatchKey>(1));

      ASSERT_TRUE(empty_set.begin() == empty_set.end());
      ASSERT_TRUE(full_set.begin() != full_set.end());

      // Increment Ops
      ASSERT_TRUE(full_set.begin() == full_set.begin()++);
      ASSERT_TRUE(full_set.begin() != ++full_set.begin());

    */
}

#[test] fn dispatch_key_set_iterator_empty() {
    todo!();
    /*
    
      DispatchKeySet empty_set;
      uint8_t i = 0;

      for (auto it = empty_set.begin(); it != empty_set.end(); ++it) {
        i++;
      }
      ASSERT_EQ(i, 0);

    */
}

#[test] fn dispatch_key_set_iterator_full() {
    todo!();
    /*
    
      DispatchKeySet full_set(DispatchKeySet::FULL);
      uint8_t i = 0;

      for (const auto& it : full_set) {
        i++;
        ASSERT_TRUE(it == static_cast<DispatchKey>(i));
        ASSERT_TRUE(it != DispatchKey::NumDispatchKeys);
      }
      ASSERT_EQ(i, static_cast<uint8_t>(DispatchKey::NumDispatchKeys) - 1);

    */
}

#[test] fn dispatch_key_set_iterator_range_full() {
    todo!();
    /*
    
      DispatchKeySet full_set(DispatchKeySet::FULL);
      uint8_t i = 0;

      for (DispatchKey dispatch_key : full_set) {
        i++;
        ASSERT_TRUE(dispatch_key == static_cast<DispatchKey>(i));
      }

      ASSERT_EQ(i, static_cast<uint8_t>(DispatchKey::NumDispatchKeys) - 1);

    */
}

#[test] fn dispatch_key_set_specific_keys() {
    todo!();
    /*
    
      DispatchKeySet keyset({
          static_cast<DispatchKey>(0), // Undefined should be ignored
          static_cast<DispatchKey>(4),
          static_cast<DispatchKey>(10),
          static_cast<DispatchKey>(15),
      });
      unordered_set<DispatchKey> visited_keys;

      for (DispatchKey key : keyset) {
        visited_keys.insert(key);
      }

      ASSERT_EQ(visited_keys.size(), 3);
      ASSERT_TRUE(
          visited_keys.find(static_cast<DispatchKey>(4)) != visited_keys.end());
      ASSERT_TRUE(
          visited_keys.find(static_cast<DispatchKey>(10)) != visited_keys.end());
      ASSERT_TRUE(
          visited_keys.find(static_cast<DispatchKey>(15)) != visited_keys.end());

    */
}

#[test] fn dispatch_key_set_fail_at_end_iterator() {
    todo!();
    /*
    
      DispatchKeySet full_set(DispatchKeySet::FULL);
      uint64_t raw_repr = full_set.raw_repr();

      EXPECT_THROW(
          DispatchKeySet::iterator(
              &raw_repr, static_cast<uint8_t>(DispatchKey::NumDispatchKeys) + 1),
          Error);

    */
}
