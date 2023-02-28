crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/test_utils.h]

enum Mode {
  Static,
  Runtime,
}

#[macro_export] macro_rules! _MAKE_TEST {
    ($TestClass:ident, $test_name:ident, $test_body:ident, $($arg:ident),*) => {
        /*
        
          TEST(TestClass, test_name) {                            
            test_body.testQ8(__VA_ARGS__);                        
          }
        */
    }
}

#[macro_export] macro_rules! _STATIC_TEST {
    ($TestClass:ident, $test_name:ident, $test_body:ident) => {
        /*
        
          _MAKE_TEST(TestClass, test_name##_static, test_body, Mode::Static)
        */
    }
}

#[macro_export] macro_rules! _RUNTIME_TEST {
    ($TestClass:ident, $test_name:ident, $test_body:ident) => {
        /*
        
          _MAKE_TEST(TestClass, test_name##_runtime, test_body, Mode::Runtime)
        */
    }
}

#[macro_export] macro_rules! _STATIC_AND_RUNTIME_TEST {
    ($TestClass:ident, $test_name:ident, $test_body:ident) => {
        /*
        
          _STATIC_TEST(TestClass, test_name, test_body)                   
          _RUNTIME_TEST(TestClass, test_name, test_body)
        */
    }
}
