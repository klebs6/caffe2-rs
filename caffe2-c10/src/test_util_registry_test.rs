crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/util/registry_test.cpp]

/**
  | Note: we use a different namespace to test if
  | the macros defined in
  |
  | Registry.h actually works with a different
  | namespace from c10.
  |
  */
pub mod c10_test {

    use super::*;

    pub struct Foo {

    }

    impl Foo {
        
        pub fn new(x: i32) -> Self {
        
            todo!();
            /*


                // LOG(INFO) << "Foo " << x;
            */
        }
    }

    c10_declare_registry!{FooRegistry, Foo, i32}
    c10_define_registry!{FooRegistry, Foo, i32}

    macro_rules! register_foo {
        ($clsname:ident) => {
            /*
                    C10_REGISTER_CLASS(FooRegistry, clsname, clsname)
            */
        }
    }

    pub struct Bar {
        base: Foo,
    }

    impl Bar {
        
        pub fn new(x: i32) -> Self {
        
            todo!();
            /*
            : foo(x),

                // LOG(INFO) << "Bar " << x;
            */
        }
    }

    register_foo!{Bar}

    pub struct AnotherBar {
        base: Foo,
    }

    impl AnotherBar {

        pub fn new(x: i32) -> Self {
        
            todo!();
            /*
            : foo(x),

                // LOG(INFO) << "AnotherBar " << x;
            */
        }
    }

    register_foo!{AnotherBar}

    #[test] fn registry_test_can_run_creator() {
        todo!();
        /*
        
              unique_ptr<Foo> bar(FooRegistry()->Create("Bar", 1));
              EXPECT_TRUE(bar != nullptr) << "Cannot create bar.";
              unique_ptr<Foo> another_bar(FooRegistry()->Create("AnotherBar", 1));
              EXPECT_TRUE(another_bar != nullptr);
            
        */
    }

    #[test] fn registry_test_return_null_on_non_existing_creator() {
        todo!();
        /*
        
          EXPECT_EQ(FooRegistry()->Create("Non-existing bar", 1), nullptr);

        */
    }

    pub fn register_foo_default()  {
        
        todo!();
            /*
                C10_REGISTER_CLASS_WITH_PRIORITY(
              FooRegistry, FooWithPriority, REGISTRY_DEFAULT, Foo);
            */
    }

    pub fn register_foo_default_again()  {
        
        todo!();
            /*
                C10_REGISTER_CLASS_WITH_PRIORITY(
              FooRegistry, FooWithPriority, REGISTRY_DEFAULT, Foo);
            */
    }

    pub fn register_foo_bar_fallback()  {
        
        todo!();
            /*
                C10_REGISTER_CLASS_WITH_PRIORITY(
              FooRegistry, FooWithPriority, REGISTRY_FALLBACK, Bar);
            */
    }

    pub fn register_foo_bar_preferred()  {
        
        todo!();
            /*
                C10_REGISTER_CLASS_WITH_PRIORITY(
              FooRegistry, FooWithPriority, REGISTRY_PREFERRED, Bar);
            */
    }

    #[test] fn registry_test_priorities() {
        todo!();
        /*
        
          FooRegistry()->SetTerminate(false);
          RegisterFooDefault();

          // throws because Foo is already registered with default priority
          // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
          EXPECT_THROW(RegisterFooDefaultAgain(), runtime_error);

        #ifdef __GXX_RTTI
          // not going to register Bar because Foo is registered with Default priority
          RegisterFooBarFallback();
          unique_ptr<Foo> bar1(FooRegistry()->Create("FooWithPriority", 1));
          EXPECT_EQ(dynamic_cast<Bar*>(bar1.get()), nullptr);

          // will register Bar because of higher priority
          RegisterFooBarPreferred();
          unique_ptr<Foo> bar2(FooRegistry()->Create("FooWithPriority", 1));
          EXPECT_NE(dynamic_cast<Bar*>(bar2.get()), nullptr);
        #endif

        */
    }
}
