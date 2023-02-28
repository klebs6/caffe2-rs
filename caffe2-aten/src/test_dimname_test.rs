crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/Dimname_test.cpp]

#[test] fn dimname_test_is_valid_identifier() {
    todo!();
    /*
    
      ASSERT_TRUE(Dimname::isValidName("a"));
      ASSERT_TRUE(Dimname::isValidName("batch"));
      ASSERT_TRUE(Dimname::isValidName("N"));
      ASSERT_TRUE(Dimname::isValidName("CHANNELS"));
      ASSERT_TRUE(Dimname::isValidName("foo_bar_baz"));
      ASSERT_TRUE(Dimname::isValidName("batch1"));
      ASSERT_TRUE(Dimname::isValidName("batch_9"));
      ASSERT_TRUE(Dimname::isValidName("_"));
      ASSERT_TRUE(Dimname::isValidName("_1"));

      ASSERT_FALSE(Dimname::isValidName(""));
      ASSERT_FALSE(Dimname::isValidName(" "));
      ASSERT_FALSE(Dimname::isValidName(" a "));
      ASSERT_FALSE(Dimname::isValidName("1batch"));
      ASSERT_FALSE(Dimname::isValidName("?"));
      ASSERT_FALSE(Dimname::isValidName("-"));
      ASSERT_FALSE(Dimname::isValidName("1"));
      ASSERT_FALSE(Dimname::isValidName("01"));

    */
}

#[test] fn dimname_test_wildcard_name() {
    todo!();
    /*
    
      Dimname wildcard = Dimname::wildcard();
      ASSERT_EQ(wildcard.type(), NameType::WILDCARD);
      ASSERT_EQ(wildcard.symbol(), Symbol::dimname("*"));

    */
}

#[test] fn dimname_test_create_normal_name() {
    todo!();
    /*
    
      auto foo = Symbol::dimname("foo");
      auto dimname = Dimname::fromSymbol(foo);
      ASSERT_EQ(dimname.type(), NameType::BASIC);
      ASSERT_EQ(dimname.symbol(), foo);
      ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname("inva.lid")), Error);
      ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname("1invalid")), Error);

    */
}

pub fn check_unify_and_match(
    dimname:  &String,
    other:    &String,
    expected: Option<String>)  {
    
    todo!();
        /*
            auto dimname1 = Dimname::fromSymbol(Symbol::dimname(dimname));
      auto dimname2 = Dimname::fromSymbol(Symbol::dimname(other));
      auto result = dimname1.unify(dimname2);
      if (expected) {
        auto expected_result = Dimname::fromSymbol(Symbol::dimname(*expected));
        ASSERT_EQ(result->symbol(), expected_result.symbol());
        ASSERT_EQ(result->type(), expected_result.type());
        ASSERT_TRUE(dimname1.matches(dimname2));
      } else {
        ASSERT_FALSE(result);
        ASSERT_FALSE(dimname1.matches(dimname2));
      }
        */
}

#[test] fn dimname_test_unify_and_match() {
    todo!();
    /*
    
      check_unify_and_match("a", "a", "a");
      check_unify_and_match("a", "*", "a");
      check_unify_and_match("*", "a", "a");
      check_unify_and_match("*", "*", "*");
      check_unify_and_match("a", "b", nullopt);

    */
}
