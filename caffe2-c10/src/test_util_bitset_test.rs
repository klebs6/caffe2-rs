crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/util/Bitset_test.cpp]

#[test] fn bitset_test_given_empty_when_getting_bit_then_is_zero() {
    todo!();
    /*
    
      bitset b;
      for (size_t i = 0; i < bitset::NUM_BITS(); ++i) {
        EXPECT_FALSE(b.get(i));
      }

    */
}

#[test] fn bitset_test_given_empty_when_unsetting_bit_then_is_zero() {
    todo!();
    /*
    
      bitset b;
      b.unset(4);
      for (size_t i = 0; i < bitset::NUM_BITS(); ++i) {
        EXPECT_FALSE(b.get(i));
      }

    */
}

#[test] fn bitset_test_given_empty_when_setting_and_unsetting_bit_then_is_zero() {
    todo!();
    /*
    
      bitset b;
      b.set(4);
      b.unset(4);
      for (size_t i = 0; i < bitset::NUM_BITS(); ++i) {
        EXPECT_FALSE(b.get(i));
      }

    */
}

#[test] fn bitset_test_given_empty_when_setting_bit_then_is_set() {
    todo!();
    /*
    
      bitset b;
      b.set(6);
      EXPECT_TRUE(b.get(6));

    */
}

#[test] fn bitset_test_given_empty_when_setting_bit_then_others_stay_unset() {
    todo!();
    /*
    
      bitset b;
      b.set(6);
      for (size_t i = 0; i < 6; ++i) {
        EXPECT_FALSE(b.get(i));
      }
      for (size_t i = 7; i < bitset::NUM_BITS(); ++i) {
        EXPECT_FALSE(b.get(i));
      }

    */
}

#[test] fn bitset_test_given_nonempty_when_setting_bit_then_is_set() {
    todo!();
    /*
    
      bitset b;
      b.set(6);
      b.set(30);
      EXPECT_TRUE(b.get(30));

    */
}

#[test] fn bitset_test_given_nonempty_when_setting_bit_then_others_stay_at_old_value() {
    todo!();
    /*
    
      bitset b;
      b.set(6);
      b.set(30);
      for (size_t i = 0; i < 6; ++i) {
        EXPECT_FALSE(b.get(i));
      }
      for (size_t i = 7; i < 30; ++i) {
        EXPECT_FALSE(b.get(i));
      }
      for (size_t i = 31; i < bitset::NUM_BITS(); ++i) {
        EXPECT_FALSE(b.get(i));
      }

    */
}

#[test] fn bitset_test_given_nonempty_when_unsetting_bit_then_is_unset() {
    todo!();
    /*
    
      bitset b;
      b.set(6);
      b.set(30);
      b.unset(6);
      EXPECT_FALSE(b.get(6));

    */
}

#[test] fn bitset_test_given_nonempty_when_unsetting_bit_then_others_stay_at_old_value() {
    todo!();
    /*
    
      bitset b;
      b.set(6);
      b.set(30);
      b.unset(6);
      for (size_t i = 0; i < 30; ++i) {
        EXPECT_FALSE(b.get(i));
      }
      EXPECT_TRUE(b.get(30));
      for (size_t i = 31; i < bitset::NUM_BITS(); ++i) {
        EXPECT_FALSE(b.get(i));
      }

    */
}

pub struct IndexCallbackMock {
    called_for_indices: Vec<usize>,
}

impl IndexCallbackMock {
    
    pub fn invoke(&mut self, index: usize)  {
        
        todo!();
        /*
            called_for_indices.push_back(index);
        */
    }
    
    pub fn expect_was_called_for_indices(&mut self, expected_indices: Vec<usize>)  {
        
        todo!();
        /*
            EXPECT_EQ(expected_indices.size(), called_for_indices.size());
        for (size_t i = 0; i < expected_indices.size(); ++i) {
          EXPECT_EQ(expected_indices[i], called_for_indices[i]);
        }
        */
    }
}

#[test] fn bitset_test_given_empty_when_calling_for_each_bit_then_doesnt_call() {
    todo!();
    /*
    
      IndexCallbackMock callback;
      bitset b;
      b.for_each_set_bit(callback);
      callback.expect_was_called_for_indices({});

    */
}

#[test] fn bitset_test_given_with_one_bit_set_when_calling_for_each_then_calls() {
    todo!();
    /*
    
      IndexCallbackMock callback;
      bitset b;
      b.set(5);
      b.for_each_set_bit(callback);
      callback.expect_was_called_for_indices({5});

    */
}

#[test] fn bitset_test_given_with_multiple_bits_set_when_calling_for_each_bit_then_calls() {
    todo!();
    /*
    
      IndexCallbackMock callback;
      bitset b;
      b.set(5);
      b.set(2);
      b.set(25);
      b.set(32);
      b.set(50);
      b.set(0);
      b.unset(25);
      b.set(10);
      b.for_each_set_bit(callback);
      callback.expect_was_called_for_indices({0, 2, 5, 10, 32, 50});

    */
}
