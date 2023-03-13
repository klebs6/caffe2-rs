crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/util/exception_test.cpp]

pub fn throw_func() -> bool {
    
    todo!();
        /*
            throw runtime_error("I'm throwing...");
        */
}

#[inline] pub fn expect_throws_eq<Functor>(
    functor:          Functor,
    expected_message: *const u8)  {

    todo!();
        /*
            try {
        forward<Functor>(functor)();
      } catch (const Error& e) {
        EXPECT_STREQ(e.what_without_backtrace(), expectedMessage);
        return;
      }
      ADD_FAILURE() << "Expected to throw exception with message \""
                    << expectedMessage << "\" but didn't throw";
        */
}

#[test] fn exception_test_torch_internal_assert_debug_only() {
    todo!();
    /*
    
    #ifdef NDEBUG
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
      ASSERT_NO_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false));
      // Does nothing - `throw_func()` should not be evaluated
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
      ASSERT_NO_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(throw_func()));
    #else
      ASSERT_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false), Error);
      ASSERT_NO_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(true));
    #endif

    */
}

#[test] fn warning_test_just_print() {
    todo!();
    /*
    
      TORCH_WARN("I'm a warning");

    */
}

#[test] fn exception_test_error_formatting() {
    todo!();
    /*
    
      expectThrowsEq(
          []() { TORCH_CHECK(false, "This is invalid"); }, "This is invalid");

      expectThrowsEq(
          []() {
            try {
              TORCH_CHECK(false, "This is invalid");
            } catch (Error& e) {
              TORCH_RETHROW(e, "While checking X");
            }
          },
          "This is invalid (While checking X)");

      expectThrowsEq(
          []() {
            try {
              try {
                TORCH_CHECK(false, "This is invalid");
              } catch (Error& e) {
                TORCH_RETHROW(e, "While checking X");
              }
            } catch (Error& e) {
              TORCH_RETHROW(e, "While checking Y");
            }
          },
          R"msg(This is invalid
      While checking X
      While checking Y)msg");

    */
}

lazy_static!{
    /*
    static int assertionArgumentCounter = 0;
    */
}

pub fn get_assertion_argument() -> i32 {
    
    todo!();
        /*
            return ++assertionArgumentCounter;
        */
}

pub fn fail_check()  {
    
    todo!();
        /*
            TORCH_CHECK(false, "message ", getAssertionArgument());
        */
}

pub fn fail_internal_assert()  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "message ", getAssertionArgument());
        */
}

#[test] fn exception_test_dont_call_argument_functions_twice_on_failure() {
    todo!();
    /*
    
      assertionArgumentCounter = 0;
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
      EXPECT_ANY_THROW(failCheck());
      EXPECT_EQ(assertionArgumentCounter, 1) << "TORCH_CHECK called argument twice";

      // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
      EXPECT_ANY_THROW(failInternalAssert());
      EXPECT_EQ(assertionArgumentCounter, 2)
          << "TORCH_INTERNAL_ASSERT called argument twice";

    */
}
