crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/test_assert.h]

#[inline] pub fn barf(
    fmt:  *const u8,
    args: &[&str])  {

    todo!();
        /*
            char msg[2048];
      va_list args;
      va_start(args, fmt);
      vsnprintf(msg, 2048, fmt, args);
      va_end(args);
      throw runtime_error(msg);
        */
}

#[macro_export] macro_rules! assert {
    ($cond:ident) => {
        /*
        
          if (AT_EXPECT(!(cond), 0)) { 
            barf("%s:%u: %s: Assertion `%s` failed.", __FILE__, __LINE__, __func__, #cond); 
          }
        */
    }
}

#[macro_export] macro_rules! try_catch_else {
    ($fn:ident, $catc:ident, $els:ident) => {
        /*
        
          {                                                             
            /* avoid mistakenly passing if els code throws exception*/  
            bool _passed = false;                                       
            try {                                                       
              fn;                                                       
              _passed = true;                                           
              els;                                                      
            } catch (const exception &e) {                         
              ASSERT(!_passed);                                         
              catc;                                                     
            }                                                           
          }
        */
    }
}

#[macro_export] macro_rules! assert_throwsm {
    ($fn:ident, $message:ident) => {
        /*
        
          TRY_CATCH_ELSE(fn, ASSERT(string(e.what()).find(message) != string::npos), ASSERT(false))
        */
    }
}

#[macro_export] macro_rules! assert_throws {
    ($fn:ident) => {
        /*
        
          ASSERT_THROWSM(fn, "");
        */
    }
}

#[macro_export] macro_rules! assert_equal {
    ($t1:ident, $t2:ident) => {
        /*
        
          ASSERT(t1.equal(t2));
        */
    }
}

/**
  | allclose broadcasts, so check same
  | size before allclose.
  |
  */
#[macro_export] macro_rules! assert_allclose {
    ($t1:ident, $t2:ident) => {
        /*
        
          ASSERT(t1.is_same_size(t2));    
          ASSERT(t1.allclose(t2));
        */
    }
}

/**
  | allclose broadcasts, so check same
  | size before allclose.
  |
  */
macro_rules! assert_allclose_tolerances {
    ($t1:ident, $t2:ident, $atol:ident, $rtol:ident) => {
        /*
        
          ASSERT(t1.is_same_size(t2));    
          ASSERT(t1.allclose(t2, atol, rtol));
        */
    }
}
