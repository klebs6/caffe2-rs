crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/List_test.cpp]

#[test] fn list_test_i_value_based_given_empty_when_calling_then_returns_true() {
    todo!();
    /*
    
        List<string> list;
        EXPECT_TRUE(list.empty());

    */
}

#[test] fn list_test_i_value_based_given_nonempty_when_calling_empty_then_returns_false() {
    todo!();
    /*
    
        List<string> list({"3"});
        EXPECT_FALSE(list.empty());

    */
}

#[test] fn list_test_i_value_based_given_empty_when_calling_size_then_returns_zero() {
    todo!();
    /*
    
        List<string> list;
        EXPECT_EQ(0, list.size());

    */
}

#[test] fn list_test_i_value_based_given_nonempty_when_calling_size_then_returns_number_of_elements() {
    todo!();
    /*
    
        List<string> list({"3", "4"});
        EXPECT_EQ(2, list.size());

    */
}

#[test] fn list_test_i_value_based_given_nonempty_when_calling_clear_then_is_empty() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      list.clear();
      EXPECT_TRUE(list.empty());

    */
}

#[test] fn list_test_i_value_based_when_calling_get_with_existing_position_then_returns_element() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      EXPECT_EQ("3", list.get(0));
      EXPECT_EQ("4", list.get(1));

    */
}

#[test] fn list_test_i_value_based_when_calling_get_with_non_existing_position_then_throws_exception() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      EXPECT_THROW(list.get(2), out_of_range);

    */
}


#[test] fn list_test_i_value_based_when_calling_extract_with_existing_position_then_returns_element() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      EXPECT_EQ("3", list.extract(0));
      EXPECT_EQ("4", list.extract(1));

    */
}

#[test] fn list_test_i_value_based_when_calling_extract_with_existing_position_then_element_becomes_invalid() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      list.extract(0);
      EXPECT_EQ("", list.get(0));

    */
}

#[test] fn list_test_i_value_based_when_calling_extract_with_non_existing_position_then_throws_exception() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      EXPECT_THROW(list.extract(2), out_of_range);

    */
}

#[test] fn list_test_i_value_based_when_calling_copying_set_with_existing_position_then_changes_element() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      string value = "5";
      list.set(1, value);
      EXPECT_EQ("3", list.get(0));
      EXPECT_EQ("5", list.get(1));

    */
}

#[test] fn list_test_i_value_based_when_calling_moving_set_with_existing_position_then_changes_element() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      string value = "5";
      list.set(1, move(value));
      EXPECT_EQ("3", list.get(0));
      EXPECT_EQ("5", list.get(1));

    */
}

#[test] fn list_test_i_value_based_when_calling_copying_set_with_non_existing_position_then_throws_exception() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      string value = "5";
      EXPECT_THROW(list.set(2, value), out_of_range);

    */
}

#[test] fn list_test_i_value_based_when_calling_moving_set_with_non_existing_position_then_throws_exception() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      string value = "5";
      EXPECT_THROW(list.set(2, move(value)), out_of_range);

    */
}

#[test] fn list_test_i_value_based_when_calling_access_operator_with_existing_position_then_returns_element() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      EXPECT_EQ("3", static_cast<string>(list[0]));
      EXPECT_EQ("4", static_cast<string>(list[1]));

    */
}

#[test] fn list_test_i_value_based_when_assigning_to_access_operator_with_existing_position_then_sets_element() {
    todo!();
    /*
    
      List<string> list({"3", "4", "5"});
      list[1] = "6";
      EXPECT_EQ("3", list.get(0));
      EXPECT_EQ("6", list.get(1));
      EXPECT_EQ("5", list.get(2));

    */
}

#[test] fn list_test_i_value_based_when_assigning_to_access_operator_from_then_sets_element() {
    todo!();
    /*
    
      List<string> list({"3", "4", "5"});
      list[1] = list[2];
      EXPECT_EQ("3", list.get(0));
      EXPECT_EQ("5", list.get(1));
      EXPECT_EQ("5", list.get(2));

    */
}

#[test] fn list_test_i_value_based_when_swapping_from_access_operator_then_swaps_elements() {
    todo!();
    /*
    
      List<string> list({"3", "4", "5"});
      swap(list[1], list[2]);
      EXPECT_EQ("3", list.get(0));
      EXPECT_EQ("5", list.get(1));
      EXPECT_EQ("4", list.get(2));

    */
}

#[test] fn list_test_i_value_based_when_calling_access_operator_with_non_existing_position_then_throws_exception() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      EXPECT_THROW(list[2], out_of_range);

    */
}

#[test] fn list_test_i_value_based_when_calling_insert_on_iterator_with_lvalue_then_inserts_element() {
    todo!();
    /*
    
      List<string> list({"3", "4", "6"});
      string v = "5";
      list.insert(list.begin() + 2, v);
      EXPECT_EQ(4, list.size());
      EXPECT_EQ("5", list.get(2));

    */
}

#[test] fn list_test_i_value_based_when_calling_insert_on_iterator_with_rvalue_then_inserts_element() {
    todo!();
    /*
    
      List<string> list({"3", "4", "6"});
      string v = "5";
      list.insert(list.begin() + 2, move(v));
      EXPECT_EQ(4, list.size());
      EXPECT_EQ("5", list.get(2));

    */
}

#[test] fn list_test_i_value_based_when_calling_insert_with_lvalue_then_returns_iterator_to_new_element() {
    todo!();
    /*
    
      List<string> list({"3", "4", "6"});
      string v = "5";
      List<string>::iterator result = list.insert(list.begin() + 2, v);
      EXPECT_EQ(list.begin() + 2, result);

    */
}

#[test] fn list_test_i_value_based_when_calling_insert_with_rvalue_then_returns_iterator_to_new_element() {
    todo!();
    /*
    
      List<string> list({"3", "4", "6"});
      string v = "5";
      List<string>::iterator result = list.insert(list.begin() + 2, move(v));
      EXPECT_EQ(list.begin() + 2, result);

    */
}

#[test] fn list_test_i_value_based_when_calling_emplace_with_lvalue_then_inserts_element() {
    todo!();
    /*
    
      List<string> list({"3", "4", "6"});
      string v = "5";
      list.emplace(list.begin() + 2, v);
      EXPECT_EQ(4, list.size());
      EXPECT_EQ("5", list.get(2));

    */
}

#[test] fn list_test_i_value_based_when_calling_emplace_with_rvalue_then_inserts_element() {
    todo!();
    /*
    
      List<string> list({"3", "4", "6"});
      string v = "5";
      list.emplace(list.begin() + 2, move(v));
      EXPECT_EQ(4, list.size());
      EXPECT_EQ("5", list.get(2));

    */
}

#[test] fn list_test_i_value_based_when_calling_emplace_with_constructor_arg_then_inserts_element() {
    todo!();
    /*
    
      List<string> list({"3", "4", "6"});
      list.emplace(list.begin() + 2, "5"); // const char* is a constructor arg to string
      EXPECT_EQ(4, list.size());
      EXPECT_EQ("5", list.get(2));

    */
}

#[test] fn list_test_i_value_based_when_calling_push_back_with_lvalue_then_inserts_element() {
    todo!();
    /*
    
      List<string> list;
      string v = "5";
      list.push_back(v);
      EXPECT_EQ(1, list.size());
      EXPECT_EQ("5", list.get(0));

    */
}

#[test] fn list_test_i_value_based_when_calling_push_back_with_rvalue_then_inserts_element() {
    todo!();
    /*
    
      List<string> list;
      string v = "5";
      list.push_back(move(v));
      EXPECT_EQ(1, list.size());
      EXPECT_EQ("5", list.get(0));

    */
}

#[test] fn list_test_i_value_based_when_calling_emplace_back_with_lvalue_then_inserts_element() {
    todo!();
    /*
    
      List<string> list;
      string v = "5";
      list.emplace_back(v);
      EXPECT_EQ(1, list.size());
      EXPECT_EQ("5", list.get(0));

    */
}

#[test] fn list_test_i_value_based_when_calling_emplace_back_with_rvalue_then_inserts_element() {
    todo!();
    /*
    
      List<string> list;
      string v = "5";
      list.emplace_back(move(v));
      EXPECT_EQ(1, list.size());
      EXPECT_EQ("5", list.get(0));

    */
}

#[test] fn list_test_i_value_based_when_calling_emplace_back_with_constructor_arg_then_inserts_element() {
    todo!();
    /*
    
      List<string> list;
      list.emplace_back("5");  // const char* is a constructor arg to string
      EXPECT_EQ(1, list.size());
      EXPECT_EQ("5", list.get(0));

    */
}

#[test] fn list_test_i_value_based_given_empty_when_iterating_then_begin_is_end() {
    todo!();
    /*
    
      List<string> list;
      const List<string> clist;
      EXPECT_EQ(list.begin(), list.end());
      EXPECT_EQ(list.begin(), list.end());
      EXPECT_EQ(clist.begin(), clist.end());
      EXPECT_EQ(clist.begin(), clist.end());

    */
}

#[test] fn list_test_i_value_based_when_iterating_then_finds_elements() {
    todo!();
    /*
    
      List<string> list({"3", "5"});
      bool found_first = false;
      bool found_second = false;
      for (List<string>::iterator iter = list.begin(); iter != list.end(); ++iter) {
        if (static_cast<string>(*iter) == "3") {
          EXPECT_FALSE(found_first);
          found_first = true;
        } else if (static_cast<string>(*iter) == "5") {
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

#[test] fn list_test_i_value_based_when_iterating_with_foreach_then_finds_elements() {
    todo!();
    /*
    
      List<string> list({"3", "5"});
      bool found_first = false;
      bool found_second = false;
      for (const string& elem : list) {
        if (elem == "3") {
          EXPECT_FALSE(found_first);
          found_first = true;
        } else if (elem == "5") {
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

#[test] fn list_test_i_value_based_given_one_element_when_erasing_then_is_empty() {
    todo!();
    /*
    
      List<string> list({"3"});
      list.erase(list.begin());
      EXPECT_TRUE(list.empty());

    */
}

#[test] fn list_test_i_value_based_given_when_erasing_then_returns_iterator() {
    todo!();
    /*
    
      List<string> list({"1", "2", "3"});
      List<string>::iterator iter = list.erase(list.begin() + 1);
      EXPECT_EQ(list.begin() + 1, iter);

    */
}

#[test] fn list_test_i_value_based_given_when_erasing_full_range_then_is_empty() {
    todo!();
    /*
    
      List<string> list({"1", "2", "3"});
      list.erase(list.begin(), list.end());
      EXPECT_TRUE(list.empty());

    */
}

#[test] fn list_test_i_value_based_when_calling_reserve_then_doesnt_crash() {
    todo!();
    /*
    
      List<string> list;
      list.reserve(100);

    */
}

#[test] fn list_test_i_value_based_when_copy_constructing_then_are_equal() {
    todo!();
    /*
    
      List<string> list1({"3", "4"});

      List<string> list2(list1);

      EXPECT_EQ(2, list2.size());
      EXPECT_EQ("3", list2.get(0));
      EXPECT_EQ("4", list2.get(1));

    */
}

#[test] fn list_test_i_value_based_when_copy_assigning_then_are_equal() {
    todo!();
    /*
    
      List<string> list1({"3", "4"});

      List<string> list2;
      list2 = list1;

      EXPECT_EQ(2, list2.size());
      EXPECT_EQ("3", list2.get(0));
      EXPECT_EQ("4", list2.get(1));

    */
}


#[test] fn list_test_i_value_based_when_copying_then_are_equal() {
    todo!();
    /*
    
      List<string> list1({"3", "4"});

      List<string> list2 = list1.copy();

      EXPECT_EQ(2, list2.size());
      EXPECT_EQ("3", list2.get(0));
      EXPECT_EQ("4", list2.get(1));

    */
}


#[test] fn list_test_i_value_based_when_move_constructing_then_new_is_correct() {
    todo!();
    /*
    
      List<string> list1({"3", "4"});

      List<string> list2(move(list1));

      EXPECT_EQ(2, list2.size());
      EXPECT_EQ("3", list2.get(0));
      EXPECT_EQ("4", list2.get(1));

    */
}


#[test] fn list_test_i_value_based_when_move_assigning_then_new_is_correct() {
    todo!();
    /*
    
      List<string> list1({"3", "4"});

      List<string> list2;
      list2 = move(list1);

      EXPECT_EQ(2, list2.size());
      EXPECT_EQ("3", list2.get(0));
      EXPECT_EQ("4", list2.get(1));

    */
}


#[test] fn list_test_i_value_based_when_move_constructing_then_old_is_empty() {
    todo!();
    /*
    
      List<string> list1({"3", "4"});

      List<string> list2(move(list1));
      EXPECT_TRUE(list1.empty());

    */
}


#[test] fn list_test_i_value_based_when_move_assigning_then_old_is_empty() {
    todo!();
    /*
    
      List<string> list1({"3", "4"});

      List<string> list2;
      list2 = move(list1);
      EXPECT_TRUE(list1.empty());

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_postfix_incrementing_then_moves_to_next_and_returns_old_position() {
    todo!();
    /*
    
      List<string> list({"3", "4"});

      List<string>::iterator iter1 = list.begin();
      List<string>::iterator iter2 = iter1++;
      EXPECT_NE("3", static_cast<string>(*iter1));
      EXPECT_EQ("3", static_cast<string>(*iter2));

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_prefix_incrementing_then_moves_to_next_and_returns_new_position() {
    todo!();
    /*
    
      List<string> list({"3", "4"});

      List<string>::iterator iter1 = list.begin();
      List<string>::iterator iter2 = ++iter1;
      EXPECT_NE("3", static_cast<string>(*iter1));
      EXPECT_NE("3", static_cast<string>(*iter2));

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_postfix_decrementing_then_moves_to_next_and_returns_old_position() {
    todo!();
    /*
    
      List<string> list({"3", "4"});

      List<string>::iterator iter1 = list.end() - 1;
      List<string>::iterator iter2 = iter1--;
      EXPECT_NE("4", static_cast<string>(*iter1));
      EXPECT_EQ("4", static_cast<string>(*iter2));

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_prefix_decrementing_then_moves_to_next_and_returns_new_position() {
    todo!();
    /*
    
      List<string> list({"3", "4"});

      List<string>::iterator iter1 = list.end() - 1;
      List<string>::iterator iter2 = --iter1;
      EXPECT_NE("4", static_cast<string>(*iter1));
      EXPECT_NE("4", static_cast<string>(*iter2));

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_increasing_then_moves_to_next_and_returns_new_position() {
    todo!();
    /*
    
      List<string> list({"3", "4", "5"});

      List<string>::iterator iter1 = list.begin();
      List<string>::iterator iter2 = iter1 += 2;
      EXPECT_EQ("5", static_cast<string>(*iter1));
      EXPECT_EQ("5", static_cast<string>(*iter2));

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_decreasing_then_moves_to_next_and_returns_new_position() {
    todo!();
    /*
    
      List<string> list({"3", "4", "5"});

      List<string>::iterator iter1 = list.end();
      List<string>::iterator iter2 = iter1 -= 2;
      EXPECT_EQ("4", static_cast<string>(*iter1));
      EXPECT_EQ("4", static_cast<string>(*iter2));

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_adding_then_returns_new() {
    todo!();
    /*
    
      List<string> list({"3", "4", "5"});

      List<string>::iterator iter1 = list.begin();
      List<string>::iterator iter2 = iter1 + 2;
      EXPECT_EQ("3", static_cast<string>(*iter1));
      EXPECT_EQ("5", static_cast<string>(*iter2));

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_subtracting_then_returns_new() {
    todo!();
    /*
    
      List<string> list({"3", "4", "5"});

      List<string>::iterator iter1 = list.end() - 1;
      List<string>::iterator iter2 = iter1 - 2;
      EXPECT_EQ("5", static_cast<string>(*iter1));
      EXPECT_EQ("3", static_cast<string>(*iter2));

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_calculating_difference_then_returns_correct_number() {
    todo!();
    /*
    
      List<string> list({"3", "4"});
      EXPECT_EQ(2, list.end() - list.begin());

    */
}


#[test] fn list_test_i_value_based_given_equal_iterators_then_are() {
    todo!();
    /*
    
      List<string> list({"3", "4"});

      List<string>::iterator iter1 = list.begin();
      List<string>::iterator iter2 = list.begin();
      EXPECT_TRUE(iter1 == iter2);
      EXPECT_FALSE(iter1 != iter2);

    */
}


#[test] fn list_test_i_value_based_given_different_iterators_then_are_not_equal() {
    todo!();
    /*
    
      List<string> list({"3", "4"});

      List<string>::iterator iter1 = list.begin();
      List<string>::iterator iter2 = list.begin();
      iter2++;

      EXPECT_FALSE(iter1 == iter2);
      EXPECT_TRUE(iter1 != iter2);

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_dereferencing_then_points_to_correct_element() {
    todo!();
    /*
    
      List<string> list({"3"});

      List<string>::iterator iter = list.begin();
      EXPECT_EQ("3", static_cast<string>(*iter));

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_assigning_new_then_changes() {
    todo!();
    /*
    
      List<string> list({"3"});

      List<string>::iterator iter = list.begin();
      *iter = "4";
      EXPECT_EQ("4", list.get(0));

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_assigning_new_from_then_changes() {
    todo!();
    /*
    
      List<string> list({"3", "4"});

      List<string>::iterator iter = list.begin();
      *iter = *(iter + 1);
      EXPECT_EQ("4", list.get(0));
      EXPECT_EQ("4", list.get(1));

    */
}


#[test] fn list_test_i_value_based_given_iterator_when_swapping_values_from_then_changes() {
    todo!();
    /*
    
      List<string> list({"3", "4"});

      List<string>::iterator iter = list.begin();
      swap(*iter, *(iter + 1));
      EXPECT_EQ("4", list.get(0));
      EXPECT_EQ("3", list.get(1));

    */
}


#[test] fn list_test_i_value_based_given_one_element_when_calling_pop_back_then_is_empty() {
    todo!();
    /*
    
      List<string> list({"3"});
      list.pop_back();
      EXPECT_TRUE(list.empty());

    */
}


#[test] fn list_test_i_value_based_given_empty_when_calling_resize_then_resizes_and_sets() {
    todo!();
    /*
    
      List<string> list;
      list.resize(2);
      EXPECT_EQ(2, list.size());
      EXPECT_EQ("", list.get(0));
      EXPECT_EQ("", list.get(1));

    */
}


#[test] fn list_test_i_value_based_given_empty_when_calling_resize_with_then_resizes_and_sets() {
    todo!();
    /*
    
      List<string> list;
      list.resize(2, "value");
      EXPECT_EQ(2, list.size());
      EXPECT_EQ("value", list.get(0));
      EXPECT_EQ("value", list.get(1));

    */
}


#[test] fn list_test_i_value_based_is_reference_type() {
    todo!();
    /*
    
      List<string> list1;
      List<string> list2(list1);
      List<string> list3;
      list3 = list1;

      list1.push_back("three");
      EXPECT_EQ(1, list1.size());
      EXPECT_EQ(1, list2.size());
      EXPECT_EQ(1, list3.size());

    */
}


#[test] fn list_test_i_value_based_copy_has_separate_storage() {
    todo!();
    /*
    
      List<string> list1;
      List<string> list2(list1.copy());
      List<string> list3;
      list3 = list1.copy();

      list1.push_back("three");
      EXPECT_EQ(1, list1.size());
      EXPECT_EQ(0, list2.size());
      EXPECT_EQ(0, list3.size());

    */
}


#[test] fn list_test_i_value_based_given_equal_lists_then_is() {
    todo!();
    /*
    
      List<string> list1({"first", "second"});
      List<string> list2({"first", "second"});

      EXPECT_EQ(list1, list2);

    */
}


#[test] fn list_test_i_value_based_given_different_lists_then_is_not_equal() {
    todo!();
    /*
    
      List<string> list1({"first", "second"});
      List<string> list2({"first", "not_second"});

      EXPECT_NE(list1, list2);

    */
}


#[test] fn list_test_non_ivalue_based_given_empty_when_calling_then_returns_true() {
    todo!();
    /*
    
        List<i64> list;
        EXPECT_TRUE(list.empty());

    */
}


#[test] fn list_test_non_ivalue_based_given_nonempty_when_calling_empty_then_returns_false() {
    todo!();
    /*
    
        List<i64> list({3});
        EXPECT_FALSE(list.empty());

    */
}


#[test] fn list_test_non_ivalue_based_given_empty_when_calling_size_then_returns_zero() {
    todo!();
    /*
    
        List<i64> list;
        EXPECT_EQ(0, list.size());

    */
}


#[test] fn list_test_non_ivalue_based_given_nonempty_when_calling_size_then_returns_number_of_elements() {
    todo!();
    /*
    
        List<i64> list({3, 4});
        EXPECT_EQ(2, list.size());

    */
}


#[test] fn list_test_non_ivalue_based_given_nonempty_when_calling_clear_then_is_empty() {
    todo!();
    /*
    
      List<i64> list({3, 4});
      list.clear();
      EXPECT_TRUE(list.empty());

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_get_with_existing_position_then_returns_element() {
    todo!();
    /*
    
      List<i64> list({3, 4});
      EXPECT_EQ(3, list.get(0));
      EXPECT_EQ(4, list.get(1));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_get_with_existing_position_then_throws_exception() {
    todo!();
    /*
    
      List<i64> list({3, 4});
      EXPECT_THROW(list.get(2), out_of_range);

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_extract_with_existing_position_then_returns_element() {
    todo!();
    /*
    
      List<i64> list({3, 4});
      EXPECT_EQ(3, list.extract(0));
      EXPECT_EQ(4, list.extract(1));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_extract_with_existing_position_then_throws_exception() {
    todo!();
    /*
    
      List<i64> list({3, 4});
      EXPECT_THROW(list.extract(2), out_of_range);

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_copying_set_with_existing_position_then_changes_element() {
    todo!();
    /*
    
      List<i64> list({3, 4});
      i64 value = 5;
      list.set(1, value);
      EXPECT_EQ(3, list.get(0));
      EXPECT_EQ(5, list.get(1));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_moving_set_with_existing_position_then_changes_element() {
    todo!();
    /*
    
      List<i64> list({3, 4});
      i64 value = 5;
      list.set(1, move(value));
      EXPECT_EQ(3, list.get(0));
      EXPECT_EQ(5, list.get(1));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_copying_set_with_existing_position_then_throws_exception() {
    todo!();
    /*
    
      List<i64> list({3, 4});
      i64 value = 5;
      EXPECT_THROW(list.set(2, value), out_of_range);

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_moving_set_with_existing_position_then_throws_exception() {
    todo!();
    /*
    
      List<i64> list({3, 4});
      i64 value = 5;
      EXPECT_THROW(list.set(2, move(value)), out_of_range);

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_access_operator_with_existing_position_then_returns_element() {
    todo!();
    /*
    
      List<i64> list({3, 4});
      EXPECT_EQ(3, static_cast<i64>(list[0]));
      EXPECT_EQ(4, static_cast<i64>(list[1]));

    */
}


#[test] fn list_test_non_ivalue_based_when_assigning_to_access_operator_with_existing_position_then_sets_element() {
    todo!();
    /*
    
      List<i64> list({3, 4, 5});
      list[1] = 6;
      EXPECT_EQ(3, list.get(0));
      EXPECT_EQ(6, list.get(1));
      EXPECT_EQ(5, list.get(2));

    */
}


#[test] fn list_test_non_ivalue_based_when_assigning_to_access_operator_from_then_sets_element() {
    todo!();
    /*
    
      List<i64> list({3, 4, 5});
      list[1] = list[2];
      EXPECT_EQ(3, list.get(0));
      EXPECT_EQ(5, list.get(1));
      EXPECT_EQ(5, list.get(2));

    */
}


#[test] fn list_test_non_ivalue_based_when_swapping_from_access_operator_then_swaps_elements() {
    todo!();
    /*
    
      List<i64> list({3, 4, 5});
      swap(list[1], list[2]);
      EXPECT_EQ(3, list.get(0));
      EXPECT_EQ(5, list.get(1));
      EXPECT_EQ(4, list.get(2));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_access_operator_with_existing_position_then_throws_exception() {
    todo!();
    /*
    
      List<i64> list({3, 4});
      EXPECT_THROW(list[2], out_of_range);

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_insert_on_iterator_with_lvalue_then_inserts_element() {
    todo!();
    /*
    
      List<i64> list({3, 4, 6});
      i64 v = 5;
      list.insert(list.begin() + 2, v);
      EXPECT_EQ(4, list.size());
      EXPECT_EQ(5, list.get(2));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_insert_on_iterator_with_rvalue_then_inserts_element() {
    todo!();
    /*
    
      List<i64> list({3, 4, 6});
      i64 v = 5;
      list.insert(list.begin() + 2, move(v));
      EXPECT_EQ(4, list.size());
      EXPECT_EQ(5, list.get(2));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_insert_with_lvalue_then_returns_iterator_to_new_element() {
    todo!();
    /*
    
      List<i64> list({3, 4, 6});
      i64 v = 5;
      List<i64>::iterator result = list.insert(list.begin() + 2, v);
      EXPECT_EQ(list.begin() + 2, result);

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_insert_with_rvalue_then_returns_iterator_to_new_element() {
    todo!();
    /*
    
      List<i64> list({3, 4, 6});
      i64 v = 5;
      List<i64>::iterator result = list.insert(list.begin() + 2, move(v));
      EXPECT_EQ(list.begin() + 2, result);

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_emplace_with_lvalue_then_inserts_element() {
    todo!();
    /*
    
      List<i64> list({3, 4, 6});
      i64 v = 5;
      list.emplace(list.begin() + 2, v);
      EXPECT_EQ(4, list.size());
      EXPECT_EQ(5, list.get(2));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_emplace_with_rvalue_then_inserts_element() {
    todo!();
    /*
    
      List<i64> list({3, 4, 6});
      i64 v = 5;
      list.emplace(list.begin() + 2, move(v));
      EXPECT_EQ(4, list.size());
      EXPECT_EQ(5, list.get(2));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_emplace_with_constructor_arg_then_inserts_element() {
    todo!();
    /*
    
      List<i64> list({3, 4, 6});
      list.emplace(list.begin() + 2, 5); // const char* is a constructor arg to i64
      EXPECT_EQ(4, list.size());
      EXPECT_EQ(5, list.get(2));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_push_back_with_lvalue_then_inserts_element() {
    todo!();
    /*
    
      List<i64> list;
      i64 v = 5;
      list.push_back(v);
      EXPECT_EQ(1, list.size());
      EXPECT_EQ(5, list.get(0));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_push_back_with_rvalue_then_inserts_element() {
    todo!();
    /*
    
      List<i64> list;
      i64 v = 5;
      list.push_back(move(v));
      EXPECT_EQ(1, list.size());
      EXPECT_EQ(5, list.get(0));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_emplace_back_with_lvalue_then_inserts_element() {
    todo!();
    /*
    
      List<i64> list;
      i64 v = 5;
      list.emplace_back(v);
      EXPECT_EQ(1, list.size());
      EXPECT_EQ(5, list.get(0));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_emplace_back_with_rvalue_then_inserts_element() {
    todo!();
    /*
    
      List<i64> list;
      i64 v = 5;
      list.emplace_back(move(v));
      EXPECT_EQ(1, list.size());
      EXPECT_EQ(5, list.get(0));

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_emplace_back_with_constructor_arg_then_inserts_element() {
    todo!();
    /*
    
      List<i64> list;
      list.emplace_back(5);  // const char* is a constructor arg to i64
      EXPECT_EQ(1, list.size());
      EXPECT_EQ(5, list.get(0));

    */
}


#[test] fn list_test_non_ivalue_based_given_empty_when_iterating_then_begin_is_end() {
    todo!();
    /*
    
      List<i64> list;
      const List<i64> clist;
      EXPECT_EQ(list.begin(), list.end());
      EXPECT_EQ(list.begin(), list.end());
      EXPECT_EQ(clist.begin(), clist.end());
      EXPECT_EQ(clist.begin(), clist.end());

    */
}


#[test] fn list_test_non_ivalue_based_when_iterating_then_finds_elements() {
    todo!();
    /*
    
      List<i64> list({3, 5});
      bool found_first = false;
      bool found_second = false;
      for (List<i64>::iterator iter = list.begin(); iter != list.end(); ++iter) {
        if (static_cast<i64>(*iter) == 3) {
          EXPECT_FALSE(found_first);
          found_first = true;
        } else if (static_cast<i64>(*iter) == 5) {
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


#[test] fn list_test_non_ivalue_based_when_iterating_with_foreach_then_finds_elements() {
    todo!();
    /*
    
      List<i64> list({3, 5});
      bool found_first = false;
      bool found_second = false;
      for (const i64& elem : list) {
        if (elem == 3) {
          EXPECT_FALSE(found_first);
          found_first = true;
        } else if (elem == 5) {
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


#[test] fn list_test_non_ivalue_based_given_one_element_when_erasing_then_is_empty() {
    todo!();
    /*
    
      List<i64> list({3});
      list.erase(list.begin());
      EXPECT_TRUE(list.empty());

    */
}


#[test] fn list_test_non_ivalue_based_given_when_erasing_then_returns_iterator() {
    todo!();
    /*
    
      List<i64> list({1, 2, 3});
      List<i64>::iterator iter = list.erase(list.begin() + 1);
      EXPECT_EQ(list.begin() + 1, iter);

    */
}


#[test] fn list_test_non_ivalue_based_given_when_erasing_full_range_then_is_empty() {
    todo!();
    /*
    
      List<i64> list({1, 2, 3});
      list.erase(list.begin(), list.end());
      EXPECT_TRUE(list.empty());

    */
}


#[test] fn list_test_non_ivalue_based_when_calling_reserve_then_doesnt_crash() {
    todo!();
    /*
    
      List<i64> list;
      list.reserve(100);

    */
}


#[test] fn list_test_non_ivalue_based_when_copy_constructing_then_are_equal() {
    todo!();
    /*
    
      List<i64> list1({3, 4});

      List<i64> list2(list1);

      EXPECT_EQ(2, list2.size());
      EXPECT_EQ(3, list2.get(0));
      EXPECT_EQ(4, list2.get(1));

    */
}


#[test] fn list_test_non_ivalue_based_when_copy_assigning_then_are_equal() {
    todo!();
    /*
    
      List<i64> list1({3, 4});

      List<i64> list2;
      list2 = list1;

      EXPECT_EQ(2, list2.size());
      EXPECT_EQ(3, list2.get(0));
      EXPECT_EQ(4, list2.get(1));

    */
}


#[test] fn list_test_non_ivalue_based_when_copying_then_are_equal() {
    todo!();
    /*
    
      List<i64> list1({3, 4});

      List<i64> list2 = list1.copy();

      EXPECT_EQ(2, list2.size());
      EXPECT_EQ(3, list2.get(0));
      EXPECT_EQ(4, list2.get(1));

    */
}


#[test] fn list_test_non_ivalue_based_when_move_constructing_then_new_is_correct() {
    todo!();
    /*
    
      List<i64> list1({3, 4});

      List<i64> list2(move(list1));

      EXPECT_EQ(2, list2.size());
      EXPECT_EQ(3, list2.get(0));
      EXPECT_EQ(4, list2.get(1));

    */
}


#[test] fn list_test_non_ivalue_based_when_move_assigning_then_new_is_correct() {
    todo!();
    /*
    
      List<i64> list1({3, 4});

      List<i64> list2;
      list2 = move(list1);

      EXPECT_EQ(2, list2.size());
      EXPECT_EQ(3, list2.get(0));
      EXPECT_EQ(4, list2.get(1));

    */
}


#[test] fn list_test_non_ivalue_based_when_move_constructing_then_old_is_empty() {
    todo!();
    /*
    
      List<i64> list1({3, 4});

      List<i64> list2(move(list1));
      EXPECT_TRUE(list1.empty());

    */
}


#[test] fn list_test_non_ivalue_based_when_move_assigning_then_old_is_empty() {
    todo!();
    /*
    
      List<i64> list1({3, 4});

      List<i64> list2;
      list2 = move(list1);
      EXPECT_TRUE(list1.empty());

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_postfix_incrementing_then_moves_to_next_and_returns_old_position() {
    todo!();
    /*
    
      List<i64> list({3, 4});

      List<i64>::iterator iter1 = list.begin();
      List<i64>::iterator iter2 = iter1++;
      EXPECT_NE(3, static_cast<i64>(*iter1));
      EXPECT_EQ(3, static_cast<i64>(*iter2));

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_prefix_incrementing_then_moves_to_next_and_returns_new_position() {
    todo!();
    /*
    
      List<i64> list({3, 4});

      List<i64>::iterator iter1 = list.begin();
      List<i64>::iterator iter2 = ++iter1;
      EXPECT_NE(3, static_cast<i64>(*iter1));
      EXPECT_NE(3, static_cast<i64>(*iter2));

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_postfix_decrementing_then_moves_to_next_and_returns_old_position() {
    todo!();
    /*
    
      List<i64> list({3, 4});

      List<i64>::iterator iter1 = list.end() - 1;
      List<i64>::iterator iter2 = iter1--;
      EXPECT_NE(4, static_cast<i64>(*iter1));
      EXPECT_EQ(4, static_cast<i64>(*iter2));

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_prefix_decrementing_then_moves_to_next_and_returns_new_position() {
    todo!();
    /*
    
      List<i64> list({3, 4});

      List<i64>::iterator iter1 = list.end() - 1;
      List<i64>::iterator iter2 = --iter1;
      EXPECT_NE(4, static_cast<i64>(*iter1));
      EXPECT_NE(4, static_cast<i64>(*iter2));

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_increasing_then_moves_to_next_and_returns_new_position() {
    todo!();
    /*
    
      List<i64> list({3, 4, 5});

      List<i64>::iterator iter1 = list.begin();
      List<i64>::iterator iter2 = iter1 += 2;
      EXPECT_EQ(5, static_cast<i64>(*iter1));
      EXPECT_EQ(5, static_cast<i64>(*iter2));

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_decreasing_then_moves_to_next_and_returns_new_position() {
    todo!();
    /*
    
      List<i64> list({3, 4, 5});

      List<i64>::iterator iter1 = list.end();
      List<i64>::iterator iter2 = iter1 -= 2;
      EXPECT_EQ(4, static_cast<i64>(*iter1));
      EXPECT_EQ(4, static_cast<i64>(*iter2));

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_adding_then_returns_new() {
    todo!();
    /*
    
      List<i64> list({3, 4, 5});

      List<i64>::iterator iter1 = list.begin();
      List<i64>::iterator iter2 = iter1 + 2;
      EXPECT_EQ(3, static_cast<i64>(*iter1));
      EXPECT_EQ(5, static_cast<i64>(*iter2));

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_subtracting_then_returns_new() {
    todo!();
    /*
    
      List<i64> list({3, 4, 5});

      List<i64>::iterator iter1 = list.end() - 1;
      List<i64>::iterator iter2 = iter1 - 2;
      EXPECT_EQ(5, static_cast<i64>(*iter1));
      EXPECT_EQ(3, static_cast<i64>(*iter2));

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_calculating_difference_then_returns_correct_number() {
    todo!();
    /*
    
      List<i64> list({3, 4});
      EXPECT_EQ(2, list.end() - list.begin());

    */
}


#[test] fn list_test_non_ivalue_based_given_equal_iterators_then_are() {
    todo!();
    /*
    
      List<i64> list({3, 4});

      List<i64>::iterator iter1 = list.begin();
      List<i64>::iterator iter2 = list.begin();
      EXPECT_TRUE(iter1 == iter2);
      EXPECT_FALSE(iter1 != iter2);

    */
}


#[test] fn list_test_non_ivalue_based_given_different_iterators_then_are_not_equal() {
    todo!();
    /*
    
      List<i64> list({3, 4});

      List<i64>::iterator iter1 = list.begin();
      List<i64>::iterator iter2 = list.begin();
      iter2++;

      EXPECT_FALSE(iter1 == iter2);
      EXPECT_TRUE(iter1 != iter2);

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_dereferencing_then_points_to_correct_element() {
    todo!();
    /*
    
      List<i64> list({3});

      List<i64>::iterator iter = list.begin();
      EXPECT_EQ(3, static_cast<i64>(*iter));

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_assigning_new_value_then_changes() {
    todo!();
    /*
    
      List<i64> list({3});

      List<i64>::iterator iter = list.begin();
      *iter = 4;
      EXPECT_EQ(4, list.get(0));

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_assigning_new_value_from_then_changes() {
    todo!();
    /*
    
      List<i64> list({3, 4});

      List<i64>::iterator iter = list.begin();
      *iter = *(iter + 1);
      EXPECT_EQ(4, list.get(0));
      EXPECT_EQ(4, list.get(1));

    */
}


#[test] fn list_test_non_ivalue_based_given_iterator_when_swapping_values_from_then_changes_value() {
    todo!();
    /*
    
      List<i64> list({3, 4});

      List<i64>::iterator iter = list.begin();
      swap(*iter, *(iter + 1));
      EXPECT_EQ(4, list.get(0));
      EXPECT_EQ(3, list.get(1));

    */
}


#[test] fn list_test_non_ivalue_based_given_one_element_when_calling_pop_back_then_is_empty() {
    todo!();
    /*
    
      List<i64> list({3});
      list.pop_back();
      EXPECT_TRUE(list.empty());

    */
}


#[test] fn list_test_non_ivalue_based_given_empty_when_calling_resize_then_resizes_and_sets_value() {
    todo!();
    /*
    
      List<i64> list;
      list.resize(2);
      EXPECT_EQ(2, list.size());
      EXPECT_EQ(0, list.get(0));
      EXPECT_EQ(0, list.get(1));

    */
}


#[test] fn list_test_non_ivalue_based_given_empty_when_calling_resize_with_value_then_resizes_and_sets() {
    todo!();
    /*
    
      List<i64> list;
      list.resize(2, 5);
      EXPECT_EQ(2, list.size());
      EXPECT_EQ(5, list.get(0));
      EXPECT_EQ(5, list.get(1));

    */
}


#[test] fn list_test_non_ivalue_based_is_reference_type() {
    todo!();
    /*
    
      List<i64> list1;
      List<i64> list2(list1);
      List<i64> list3;
      list3 = list1;

      list1.push_back(3);
      EXPECT_EQ(1, list1.size());
      EXPECT_EQ(1, list2.size());
      EXPECT_EQ(1, list3.size());

    */
}


#[test] fn list_test_non_ivalue_based_copy_has_separate_storage() {
    todo!();
    /*
    
      List<i64> list1;
      List<i64> list2(list1.copy());
      List<i64> list3;
      list3 = list1.copy();

      list1.push_back(3);
      EXPECT_EQ(1, list1.size());
      EXPECT_EQ(0, list2.size());
      EXPECT_EQ(0, list3.size());

    */
}


#[test] fn list_test_non_ivalue_based_given_equal_lists_then_is() {
    todo!();
    /*
    
      List<i64> list1({1, 3});
      List<i64> list2({1, 3});

      EXPECT_EQ(list1, list2);

    */
}


#[test] fn list_test_non_ivalue_based_given_different_lists_then_is_not_equal() {
    todo!();
    /*
    
      List<i64> list1({1, 3});
      List<i64> list2({1, 2});

      EXPECT_NE(list1, list2);

    */
}


#[test] fn list_test_non_ivalue_based_is_checks_identity() {
    todo!();
    /*
    
      List<i64> list1({1, 3});
      const auto list2 = list1;

      EXPECT_TRUE(list1.is(list2));

    */
}


#[test] fn list_test_non_ivalue_based_same_value_different_storage_then_is_returns_false() {
    todo!();
    /*
    
      List<i64> list1({1, 3});
      const auto list2 = list1.copy();

      EXPECT_FALSE(list1.is(list2));

    */
}


#[test] fn list_test_can_access_string_by_reference() {
    todo!();
    /*
    
      List<string> list({"one", "two"});
      const auto& listRef = list;
      static_assert(is_same<decltype(listRef[1]), const string&>::value,
                    "const List<string> acccess should be by const reference");
      string str = list[1];
      const string& strRef = listRef[1];
      EXPECT_EQ("two", str);
      EXPECT_EQ("two", strRef);

    */
}


#[test] fn list_test_can_access_optional_string_by_reference() {
    todo!();
    /*
    
      List<optional<string>> list({"one", "two", nullopt});
      const auto& listRef = list;
      static_assert(
          is_same<decltype(listRef[1]), optional<reference_wrapper<const string>>>::value,
          "List<optional<string>> acccess should be by const reference");
      optional<string> str1 = list[1];
      optional<string> str2 = list[2];
      decltype(auto) strRef1 = listRef[1];
      decltype(auto) strRef2 = listRef[2];
      EXPECT_EQ("two", str1.value());
      EXPECT_FALSE(str2.has_value());
      EXPECT_EQ("two", strRef1.value().get());
      EXPECT_FALSE(strRef2.has_value());

    */
}


#[test] fn list_test_can_access_tensor_by_reference() {
    todo!();
    /*
    
      List<Tensor> list;
      const auto& listRef = list;
      static_assert(
          is_same<decltype(listRef[0]), const Tensor&>::value,
          "List<Tensor> access should be by const reference");

    */
}
