crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/util/logging_test.cpp]

#[test] fn logging_test_enforce_true() {
    todo!();
    /*
    
      // This should just work.
      CAFFE_ENFORCE(true, "Isn't it?");

    */
}

#[test] fn logging_test_enforce_false() {
    todo!();
    /*
    
      bool kFalse = false;
      swap(FLAGS_caffe2_use_fatal_for_enforce, kFalse);
      try {
        CAFFE_ENFORCE(false, "This throws.");
        // This should never be triggered.
        ADD_FAILURE();
      } catch (const ::Error&) {
      }
      swap(FLAGS_caffe2_use_fatal_for_enforce, kFalse);

    */
}

#[test] fn logging_test_enforce_equals() {
    todo!();
    /*
    
      int x = 4;
      int y = 5;
      int z = 0;
      try {
        CAFFE_ENFORCE_THAT(equal_to<void>(), ==, ++x, ++y, "Message: ", z++);
        // This should never be triggered.
        ADD_FAILURE();
      } catch (const ::Error& err) {
        auto errStr = string(err.what());
        EXPECT_NE(errStr.find("5 vs 6"), string::npos);
        EXPECT_NE(errStr.find("Message: 0"), string::npos);
      }

      // arguments are expanded only once
      CAFFE_ENFORCE_THAT(equal_to<void>(), ==, ++x, y);
      EXPECT_EQ(x, 6);
      EXPECT_EQ(y, 6);
      EXPECT_EQ(z, 1);

    */
}

pub struct EnforceEqWithCaller {

}

impl EnforceEqWithCaller {
    
    pub fn test(&mut self, x: *const u8)  {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ_WITH_CALLER(1, 1, "variable: ", x, " is a variable");
        */
    }
}

#[test] fn logging_test_enforce_message_variables() {
    todo!();
    /*
    
      const char* const x = "hello";
      CAFFE_ENFORCE_EQ(1, 1, "variable: ", x, " is a variable");

      EnforceEqWithCaller e;
      e.test(x);

    */
}

#[test] fn logging_test_enforce_equals_object_with_reference_to_temporary_without_use_out_of_scope() {
    todo!();
    /*
    
      vector<int> x = {1, 2, 3, 4};
      // This case is a little tricky. We have a temporary
      // initializer_list to which our temporary ArrayRef
      // refers. Temporary lifetime extension by binding a const reference
      // to the ArrayRef doesn't extend the lifetime of the
      // initializer_list, just the ArrayRef, so we end up with a
      // dangling ArrayRef. This test forces the implementation to get it
      // right.
      CAFFE_ENFORCE_EQ(x, (&[int]{1, 2, 3, 4}));

    */
}

#[derive(PartialEq,Eq)]
pub struct Noncopyable {
    x: i32,
}

impl Noncopyable {
    
    pub fn new(a: i32) -> Self {
    
        todo!();
        /*
        : x(a),

        
        */
    }
}

impl fmt::Display for Noncopyable {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            out << "Noncopyable(" << nc.x << ")";
      return out;
        */
    }
}

#[test] fn logging_test_doesnt_copy_compared_objects() {
    todo!();
    /*
    
      CAFFE_ENFORCE_EQ(Noncopyable(123), Noncopyable(123));

    */
}


#[test] fn logging_test_enforce_showcase() {
    todo!();
    /*
    
      // It's not really a test but rather a convenient thing that you can run and
      // see all messages
      int one = 1;
      int two = 2;
      int three = 3;
    #define WRAP_AND_PRINT(exp)                    \
      try {                                        \
        exp;                                       \
      } catch (const ::Error&) {              \
        /* ::Error already does LOG(ERROR) */ \
      }
      WRAP_AND_PRINT(CAFFE_ENFORCE_EQ(one, two));
      WRAP_AND_PRINT(CAFFE_ENFORCE_NE(one * 2, two));
      WRAP_AND_PRINT(CAFFE_ENFORCE_GT(one, two));
      WRAP_AND_PRINT(CAFFE_ENFORCE_GE(one, two));
      WRAP_AND_PRINT(CAFFE_ENFORCE_LT(three, two));
      WRAP_AND_PRINT(CAFFE_ENFORCE_LE(three, two));

      WRAP_AND_PRINT(CAFFE_ENFORCE_EQ(
          one * two + three, three * two, "It's a pretty complicated expression"));

      WRAP_AND_PRINT(CAFFE_ENFORCE_THAT(
          equal_to<void>(), ==, one * two + three, three * two));

    */
}

#[test] fn logging_test_join() {
    todo!();
    /*
    
      auto s = Join(", ", vector<int>({1, 2, 3}));
      EXPECT_EQ(s, "1, 2, 3");
      s = Join(":", vector<string>());
      EXPECT_EQ(s, "");
      s = Join(", ", set<int>({3, 1, 2}));
      EXPECT_EQ(s, "1, 2, 3");

    */
}

#[test] fn logging_test_dangling_else() {
    todo!();
    /*
    
      if (true)
        DCHECK_EQ(1, 1);
      else
        GTEST_FAIL();

    */
}

#[cfg(GTEST_HAS_DEATH_TEST)]
#[test] fn logging_death_test_enforce_using_fatal() {
    todo!();
    /*
    
      bool kTrue = true;
      swap(FLAGS_caffe2_use_fatal_for_enforce, kTrue);
      EXPECT_DEATH(CAFFE_ENFORCE(false, "This goes fatal."), "");
      swap(FLAGS_caffe2_use_fatal_for_enforce, kTrue);

    */
}
