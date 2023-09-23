crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/Dict_test.cpp]

#[test] fn dict_test_given_empty_when_calling_then_returns_true() {
    todo!();
    /*
    
        Dict<i64, string> dict;
        EXPECT_TRUE(dict.empty());

    */
}

#[test] fn dict_test_given_nonempty_when_calling_empty_then_returns_false() {
    todo!();
    /*
    
        Dict<i64, string> dict;
        dict.insert(3, "value");
        EXPECT_FALSE(dict.empty());

    */
}

#[test] fn dict_test_given_empty_when_calling_size_then_returns_zero() {
    todo!();
    /*
    
        Dict<i64, string> dict;
        EXPECT_EQ(0, dict.size());

    */
}

#[test] fn dict_test_given_nonempty_when_calling_size_then_returns_number_of_elements() {
    todo!();
    /*
    
        Dict<i64, string> dict;
        dict.insert(3, "value");
        dict.insert(4, "value2");
        EXPECT_EQ(2, dict.size());

    */
}

#[test] fn dict_test_given_nonempty_when_calling_clear_then_is_empty() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "value");
      dict.insert(4, "value2");
      dict.clear();
      EXPECT_TRUE(dict.empty());

    */
}

#[test] fn dict_test_when_inserting_new_key_then_returns_true_and_iterator_to_element() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      pair<Dict<i64, string>::iterator, bool> result = dict.insert(3, "value");
      EXPECT_TRUE(result.second);
      EXPECT_EQ(3, result.first->key());
      EXPECT_EQ("value", result.first->value());

    */
}

#[test] fn dict_test_when_inserting_existing_key_then_returns_false_and_iterator_to_element() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "old_value");
      pair<Dict<i64, string>::iterator, bool> result = dict.insert(3, "new_value");
      EXPECT_FALSE(result.second);
      EXPECT_EQ(3, result.first->key());
      EXPECT_EQ("old_value", result.first->value());

    */
}

#[test] fn dict_test_when_inserting_existing_key_then_does_not_modify() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "old_value");
      dict.insert(3, "new_value");
      EXPECT_EQ(1, dict.size());
      EXPECT_EQ(3, dict.begin()->key());
      EXPECT_EQ("old_value", dict.begin()->value());

    */
}

#[test] fn dict_test_when_insert_or_assigning_new_key_then_returns_true_and_iterator_to_element() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      pair<Dict<i64, string>::iterator, bool> result = dict.insert_or_assign(3, "value");
      EXPECT_TRUE(result.second);
      EXPECT_EQ(3, result.first->key());
      EXPECT_EQ("value", result.first->value());

    */
}

#[test] fn dict_test_when_insert_or_assigning_existing_key_then_returns_false_and_iterator_to_changed_element() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "old_value");
      pair<Dict<i64, string>::iterator, bool> result = dict.insert_or_assign(3, "new_value");
      EXPECT_FALSE(result.second);
      EXPECT_EQ(3, result.first->key());
      EXPECT_EQ("new_value", result.first->value());

    */
}

#[test] fn dict_test_when_insert_or_assigning_existing_key_then_does_modify() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "old_value");
      dict.insert_or_assign(3, "new_value");
      EXPECT_EQ(1, dict.size());
      EXPECT_EQ(3, dict.begin()->key());
      EXPECT_EQ("new_value", dict.begin()->value());

    */
}

#[test] fn dict_test_given_empty_when_iterating_then_begin_is_end() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      EXPECT_EQ(dict.begin(), dict.end());

    */
}

#[test] fn dict_test_given_mutable_when_iterating_then_finds_elements() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(5, "5");
      bool found_first = false;
      bool found_second = false;
      for (Dict<i64, string>::iterator iter = dict.begin(); iter != dict.end(); ++iter) {
        if (iter->key() == 3) {
          EXPECT_EQ("3", iter->value());
          EXPECT_FALSE(found_first);
          found_first = true;
        } else if (iter->key() == 5) {
          EXPECT_EQ("5", iter->value());
          EXPECT_FALSE(found_second);
          found_second = true;
        } else {
          ADD_FAILURE();
        }
      }
      EXPECT_TRUE(found_first);
      EXPECT_TRUE(found_second);

    */
}

#[test] fn dict_test_given_mutable_when_iterating_with_foreach_then_finds_elements() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(5, "5");
      bool found_first = false;
      bool found_second = false;
      for (const auto& elem : dict) {
        if (elem.key() == 3) {
          EXPECT_EQ("3", elem.value());
          EXPECT_FALSE(found_first);
          found_first = true;
        } else if (elem.key() == 5) {
          EXPECT_EQ("5", elem.value());
          EXPECT_FALSE(found_second);
          found_second = true;
        } else {
          ADD_FAILURE();
        }
      }
      EXPECT_TRUE(found_first);
      EXPECT_TRUE(found_second);

    */
}

#[test] fn dict_test_given_const_when_iterating_then_finds_elements() {
    todo!();
    /*
    
      Dict<i64, string> dict_;
      dict_.insert(3, "3");
      dict_.insert(5, "5");
      const Dict<i64, string>& dict = dict_;
      bool found_first = false;
      bool found_second = false;
      for (Dict<i64, string>::iterator iter = dict.begin(); iter != dict.end(); ++iter) {
        if (iter->key() == 3) {
          EXPECT_EQ("3", iter->value());
          EXPECT_FALSE(found_first);
          found_first = true;
        } else if (iter->key() == 5) {
          EXPECT_EQ("5", iter->value());
          EXPECT_FALSE(found_second);
          found_second = true;
        } else {
          ADD_FAILURE();
        }
      }
      EXPECT_TRUE(found_first);
      EXPECT_TRUE(found_second);

    */
}

#[test] fn dict_test_given_const_when_iterating_with_foreach_then_finds_elements() {
    todo!();
    /*
    
      Dict<i64, string> dict_;
      dict_.insert(3, "3");
      dict_.insert(5, "5");
      const Dict<i64, string>& dict = dict_;
      bool found_first = false;
      bool found_second = false;
      for (const auto& elem : dict) {
        if (elem.key() == 3) {
          EXPECT_EQ("3", elem.value());
          EXPECT_FALSE(found_first);
          found_first = true;
        } else if (elem.key() == 5) {
          EXPECT_EQ("5", elem.value());
          EXPECT_FALSE(found_second);
          found_second = true;
        } else {
          ADD_FAILURE();
        }
      }
      EXPECT_TRUE(found_first);
      EXPECT_TRUE(found_second);

    */
}

#[test] fn dict_test_given_iterator_then_can_modify_value() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "old_value");
      dict.begin()->setValue("new_value");
      EXPECT_EQ("new_value", dict.begin()->value());

    */
}

#[test] fn dict_test_given_one_element_when_erasing_by_iterator_then_is_empty() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.erase(dict.begin());
      EXPECT_TRUE(dict.empty());

    */
}

#[test] fn dict_test_given_one_element_when_erasing_by_key_then_returns_and_is_empty() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      bool result = dict.erase(3);
      EXPECT_EQ(1, result);
      EXPECT_TRUE(dict.empty());

    */
}

#[test] fn dict_test_given_one_element_when_erasing_by_nonexisting_key_then_returns_zero_and_is_unchanged() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      bool result = dict.erase(4);
      EXPECT_EQ(0, result);
      EXPECT_EQ(1, dict.size());

    */
}

#[test] fn dict_test_when_calling_at_with_existing_key_then_returns_correct_element() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(4, "4");
      EXPECT_EQ("4", dict.at(4));

    */
}

#[test] fn dict_test_when_calling_at_with_non_existing_key_then_returns_correct_element() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(4, "4");
      EXPECT_THROW(dict.at(5), out_of_range);

    */
}

#[test] fn dict_test_given_mutable_when_calling_find_on_existing_key_then_finds_correct_element() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(4, "4");
      Dict<i64, string>::iterator found = dict.find(3);
      EXPECT_EQ(3, found->key());
      EXPECT_EQ("3", found->value());

    */
}

#[test] fn dict_test_given_mutable_when_calling_find_on_non_existing_key_then_returns_end() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(4, "4");
      Dict<i64, string>::iterator found = dict.find(5);
      EXPECT_EQ(dict.end(), found);

    */
}

#[test] fn dict_test_given_const_when_calling_find_on_existing_key_then_finds_correct_element() {
    todo!();
    /*
    
      Dict<i64, string> dict_;
      dict_.insert(3, "3");
      dict_.insert(4, "4");
      const Dict<i64, string>& dict = dict_;
      Dict<i64, string>::iterator found = dict.find(3);
      EXPECT_EQ(3, found->key());
      EXPECT_EQ("3", found->value());

    */
}

#[test] fn dict_test_given_const_when_calling_find_on_non_existing_key_then_returns_end() {
    todo!();
    /*
    
      Dict<i64, string> dict_;
      dict_.insert(3, "3");
      dict_.insert(4, "4");
      const Dict<i64, string>& dict = dict_;
      Dict<i64, string>::iterator found = dict.find(5);
      EXPECT_EQ(dict.end(), found);

    */
}

#[test] fn dict_test_when_calling_contains_with_existing_key_then_returns_true() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(4, "4");
      EXPECT_TRUE(dict.contains(3));

    */
}

#[test] fn dict_test_when_calling_contains_with_non_existing_key_then_returns_false() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(4, "4");
      EXPECT_FALSE(dict.contains(5));

    */
}

#[test] fn dict_test_when_calling_reserve_then_doesnt_crash() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.reserve(100);

    */
}

#[test] fn dict_test_when_copy_constructing_then_are_equal() {
    todo!();
    /*
    
      Dict<i64, string> dict1;
      dict1.insert(3, "3");
      dict1.insert(4, "4");

      Dict<i64, string> dict2(dict1);

      EXPECT_EQ(2, dict2.size());
      EXPECT_EQ("3", dict2.at(3));
      EXPECT_EQ("4", dict2.at(4));

    */
}

#[test] fn dict_test_when_copy_assigning_then_are_equal() {
    todo!();
    /*
    
      Dict<i64, string> dict1;
      dict1.insert(3, "3");
      dict1.insert(4, "4");

      Dict<i64, string> dict2;
      dict2 = dict1;

      EXPECT_EQ(2, dict2.size());
      EXPECT_EQ("3", dict2.at(3));
      EXPECT_EQ("4", dict2.at(4));

    */
}

#[test] fn dict_test_when_copying_then_are_equal() {
    todo!();
    /*
    
      Dict<i64, string> dict1;
      dict1.insert(3, "3");
      dict1.insert(4, "4");

      Dict<i64, string> dict2 = dict1.copy();

      EXPECT_EQ(2, dict2.size());
      EXPECT_EQ("3", dict2.at(3));
      EXPECT_EQ("4", dict2.at(4));

    */
}

#[test] fn dict_test_when_move_constructing_then_new_is_correct() {
    todo!();
    /*
    
      Dict<i64, string> dict1;
      dict1.insert(3, "3");
      dict1.insert(4, "4");

      Dict<i64, string> dict2(move(dict1));

      EXPECT_EQ(2, dict2.size());
      EXPECT_EQ("3", dict2.at(3));
      EXPECT_EQ("4", dict2.at(4));

    */
}

#[test] fn dict_test_when_move_assigning_then_new_is_correct() {
    todo!();
    /*
    
      Dict<i64, string> dict1;
      dict1.insert(3, "3");
      dict1.insert(4, "4");

      Dict<i64, string> dict2;
      dict2 = move(dict1);

      EXPECT_EQ(2, dict2.size());
      EXPECT_EQ("3", dict2.at(3));
      EXPECT_EQ("4", dict2.at(4));

    */
}

#[test] fn dict_test_when_move_constructing_then_old_is_empty() {
    todo!();
    /*
    
      Dict<i64, string> dict1;
      dict1.insert(3, "3");
      dict1.insert(4, "4");

      Dict<i64, string> dict2(move(dict1));
      EXPECT_TRUE(dict1.empty());

    */
}

#[test] fn dict_test_when_move_assigning_then_old_is_empty() {
    todo!();
    /*
    
      Dict<i64, string> dict1;
      dict1.insert(3, "3");
      dict1.insert(4, "4");

      Dict<i64, string> dict2;
      dict2 = move(dict1);
      EXPECT_TRUE(dict1.empty());

    */
}

#[test] fn dict_test_given_iterator_when_postfix_incrementing_then_moves_to_next_and_returns_old_position() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(4, "4");

      Dict<i64, string>::iterator iter1 = dict.begin();
      Dict<i64, string>::iterator iter2 = iter1++;
      EXPECT_NE(dict.begin()->key(), iter1->key());
      EXPECT_EQ(dict.begin()->key(), iter2->key());

    */
}

#[test] fn dict_test_given_iterator_when_prefix_incrementing_then_moves_to_next_and_returns_new_position() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(4, "4");

      Dict<i64, string>::iterator iter1 = dict.begin();
      Dict<i64, string>::iterator iter2 = ++iter1;
      EXPECT_NE(dict.begin()->key(), iter1->key());
      EXPECT_NE(dict.begin()->key(), iter2->key());

    */
}

#[test] fn dict_test_given_equal_iterators_then_are() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(4, "4");

      Dict<i64, string>::iterator iter1 = dict.begin();
      Dict<i64, string>::iterator iter2 = dict.begin();
      EXPECT_TRUE(iter1 == iter2);
      EXPECT_FALSE(iter1 != iter2);

    */
}

#[test] fn dict_test_given_different_iterators_then_are_not_equal() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(4, "4");

      Dict<i64, string>::iterator iter1 = dict.begin();
      Dict<i64, string>::iterator iter2 = dict.begin();
      iter2++;

      EXPECT_FALSE(iter1 == iter2);
      EXPECT_TRUE(iter1 != iter2);

    */
}

#[test] fn dict_test_given_iterator_when_dereferencing_then_points_to_correct_element() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");

      Dict<i64, string>::iterator iter = dict.begin();
      EXPECT_EQ(3, (*iter).key());
      EXPECT_EQ("3", (*iter).value());
      EXPECT_EQ(3, iter->key());
      EXPECT_EQ("3", iter->value());

    */
}

#[test] fn dict_test_given_iterator_when_writing_to_value_then_changes() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");

      Dict<i64, string>::iterator iter = dict.begin();

      (*iter).setValue("new_value");
      EXPECT_EQ("new_value", dict.begin()->value());

      iter->setValue("new_value_2");
      EXPECT_EQ("new_value_2", dict.begin()->value());

    */
}

#[test] fn list_test_i_value_based_given_iterator_when_writing_to_from_then_changes() {
    todo!();
    /*
    
      Dict<i64, string> dict;
      dict.insert(3, "3");
      dict.insert(4, "4");
      dict.insert(5, "5");

      (*dict.find(3)).setValue(dict.find(4)->value());
      EXPECT_EQ("4", dict.find(3)->value());

      dict.find(3)->setValue(dict.find(5)->value());
      EXPECT_EQ("5", dict.find(3)->value());

    */
}

#[test] fn dict_test_is_reference_type() {
    todo!();
    /*
    
      Dict<i64, string> dict1;
      Dict<i64, string> dict2(dict1);
      Dict<i64, string> dict3;
      dict3 = dict1;

      dict1.insert(3, "three");
      EXPECT_EQ(1, dict1.size());
      EXPECT_EQ(1, dict2.size());
      EXPECT_EQ(1, dict3.size());

    */
}

#[test] fn dict_test_copy_has_separate_storage() {
    todo!();
    /*
    
      Dict<i64, string> dict1;
      Dict<i64, string> dict2(dict1.copy());
      Dict<i64, string> dict3;
      dict3 = dict1.copy();

      dict1.insert(3, "three");
      EXPECT_EQ(1, dict1.size());
      EXPECT_EQ(0, dict2.size());
      EXPECT_EQ(0, dict3.size());

    */
}

#[test] fn dict_test_tensor_as_key() {
    todo!();
    /*
    
      Dict<Tensor, string> dict;
      Tensor key1 = tensor(3);
      Tensor key2 = tensor(4);
      dict.insert(key1, "three");
      dict.insert(key2, "four");

      EXPECT_EQ(2, dict.size());

      Dict<Tensor, string>::iterator found_key1 = dict.find(key1);
      ASSERT_EQUAL(key1, found_key1->key());
      EXPECT_EQ("three", found_key1->value());

      Dict<Tensor, string>::iterator found_nokey1 = dict.find(tensor(3));
      Dict<Tensor, string>::iterator found_nokey2 = dict.find(tensor(5));
      EXPECT_EQ(dict.end(), found_nokey1);
      EXPECT_EQ(dict.end(), found_nokey2);

    */
}

#[test] fn dict_test_equality() {
    todo!();
    /*
    
      Dict<string, i64> dict;
      dict.insert("one", 1);
      dict.insert("two", 2);

      Dict<string, i64> dictSameValue;
      dictSameValue.insert("one", 1);
      dictSameValue.insert("two", 2);

      Dict<string, i64> dictNotEqual;
      dictNotEqual.insert("foo", 1);
      dictNotEqual.insert("bar", 2);

      Dict<string, i64> dictRef = dict;

      EXPECT_EQ(dict, dictSameValue);
      EXPECT_NE(dict, dictNotEqual);
      EXPECT_NE(dictSameValue, dictNotEqual);
      EXPECT_FALSE(dict.is(dictSameValue));
      EXPECT_TRUE(dict.is(dictRef));

    */
}
