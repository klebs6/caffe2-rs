crate::ix!();

lazy_static!{
    static ref gTestInitFunctionHasBeenRun:     bool = false;
    static ref gTestFailInitFunctionHasBeenRun: bool = false;
    /*
       int dummy_argc = 1;
       const char* dummy_name = "foo";
       char** dummy_argv = const_cast<char**>(&dummy_name);
       */
}

pub fn test_init_function(a: *mut i32, b: *mut *mut *mut u8) -> bool {
    todo!();
    /*
  gTestInitFunctionHasBeenRun = true;
  return true;
  */
}

pub fn test_fail_init_function(a: *mut i32, b: *mut *mut *mut u8) -> bool {
    todo!();
    /*
  gTestFailInitFunctionHasBeenRun = true;
  return false;
    */
}

register_caffe2_init_function!{
    test_init_function,
    test_init_function,
    "Just a test to see if GlobalInit invokes registered functions correctly."
}

#[test] fn InitTest_TestInitFunctionHasRun() {
    todo!();
    /*
      caffe2::GlobalInit(&dummy_argc, &dummy_argv);
      EXPECT_TRUE(gTestInitFunctionHasBeenRun);
      EXPECT_FALSE(gTestFailInitFunctionHasBeenRun);
  */
}

#[test] fn InitTest_CanRerunGlobalInit() {
    todo!();
    /*
      caffe2::GlobalInit(&dummy_argc, &dummy_argv);
      EXPECT_TRUE(caffe2::GlobalInit(&dummy_argc, &dummy_argv));
  */
}

#[inline] pub fn late_register_init_function()  {
    
    todo!();
    /*
        ::caffe2::InitRegisterer testInitFunc(
          TestInitFunction, false, "This should succeed but warn");
    */
}

#[inline] pub fn late_register_early_init_function()  {
    
    todo!();
    /*
        ::caffe2::InitRegisterer testSecondInitFunc(
          TestInitFunction, true, "This should fail for early init");
    */
}

#[inline] pub fn late_register_fail_init_function()  {
    
    todo!();
    /*
        ::caffe2::InitRegisterer testSecondInitFunc(
          TestFailInitFunction, false, "This should fail for failed init");
    */
}

#[test] fn InitTest_FailLateRegisterInitFunction() {
    todo!();
    /*
      caffe2::GlobalInit(&dummy_argc, &dummy_argv);
      LateRegisterInitFunction();
      EXPECT_THROW(LateRegisterEarlyInitFunction(), ::c10::Error);
      EXPECT_THROW(LateRegisterFailInitFunction(), ::c10::Error);
      EXPECT_TRUE(gTestInitFunctionHasBeenRun);
      EXPECT_TRUE(gTestFailInitFunctionHasBeenRun);
  */
}
