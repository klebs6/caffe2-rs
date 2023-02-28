crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/util/ThreadLocal_test.cpp]

#[test] fn thread_local_test_no_op_scope_with_one_var() {
    todo!();
    /*
    
      C10_DEFINE_TLS_static(string, str);

    */
}

#[test] fn thread_local_test_no_op_scope_with_two_vars() {
    todo!();
    /*
    
      C10_DEFINE_TLS_static(string, str);
      C10_DEFINE_TLS_static(string, str2);

    */
}

#[test] fn thread_local_test_scope_with_one_var() {
    todo!();
    /*
    
      C10_DEFINE_TLS_static(string, str);
      EXPECT_EQ(*str, string());
      EXPECT_EQ(*str, "");

      *str = "abc";
      EXPECT_EQ(*str, "abc");
      EXPECT_EQ(str->length(), 3);
      EXPECT_EQ(str.get(), "abc");

    */
}

#[test] fn thread_local_test_scope_with_two_vars() {
    todo!();
    /*
    
      C10_DEFINE_TLS_static(string, str);
      EXPECT_EQ(*str, "");

      C10_DEFINE_TLS_static(string, str2);

      *str = "abc";
      EXPECT_EQ(*str, "abc");
      EXPECT_EQ(*str2, "");

      *str2 = *str;
      EXPECT_EQ(*str, "abc");
      EXPECT_EQ(*str2, "abc");

      str->clear();
      EXPECT_EQ(*str, "");
      EXPECT_EQ(*str2, "abc");

    */
}

#[test] fn thread_local_test_inner_scope_with_two_vars() {
    todo!();
    /*
    
      C10_DEFINE_TLS_static(string, str);
      *str = "abc";

      {
        C10_DEFINE_TLS_static(string, str2);
        EXPECT_EQ(*str2, "");

        *str2 = *str;
        EXPECT_EQ(*str, "abc");
        EXPECT_EQ(*str2, "abc");

        str->clear();
        EXPECT_EQ(*str2, "abc");
      }

      EXPECT_EQ(*str, "");

    */
}

pub struct Foo {
}

lazy_static!{
    /*
    c10_define_tls_class_static!{Foo, string, str_}
    c10_define_tls_static!{string, global_}
    c10_define_tls_static!{string, global2_}
    c10_define_tls_static!{string, global3_}
    c10_define_tls_static!{string, global4_}
    */
}

#[test] fn thread_local_test_class_scope() {
    todo!();
    /*
    
      EXPECT_EQ(*Foo::str_, "");

      *Foo::str_ = "abc";
      EXPECT_EQ(*Foo::str_, "abc");
      EXPECT_EQ(Foo::str_->length(), 3);
      EXPECT_EQ(Foo::str_.get(), "abc");

    */
}

#[test] fn thread_local_test_two_global_scope_vars() {
    todo!();
    /*
    
      EXPECT_EQ(*global_, "");
      EXPECT_EQ(*global2_, "");

      *global_ = "abc";
      EXPECT_EQ(global_->length(), 3);
      EXPECT_EQ(*global_, "abc");
      EXPECT_EQ(*global2_, "");

      *global2_ = *global_;
      EXPECT_EQ(*global_, "abc");
      EXPECT_EQ(*global2_, "abc");

      global_->clear();
      EXPECT_EQ(*global_, "");
      EXPECT_EQ(*global2_, "abc");
      EXPECT_EQ(global2_.get(), "abc");

    */
}

#[test] fn thread_local_test_global_with_scope_vars() {
    todo!();
    /*
    
      *global3_ = "abc";

      C10_DEFINE_TLS_static(string, str);

      swap(*global3_, *str);
      EXPECT_EQ(*str, "abc");
      EXPECT_EQ(*global3_, "");

    */
}

#[test] fn thread_local_test_with_scope_var() {
    todo!();
    /*
    
      C10_DEFINE_TLS_static(string, str);
      *str = "abc";

      atomic_bool b(false);
      thread t([&b]() {
        EXPECT_EQ(*str, "");
        *str = "def";
        b = true;
        EXPECT_EQ(*str, "def");
      });
      t.join();

      EXPECT_TRUE(b);
      EXPECT_EQ(*str, "abc");

    */
}

#[test] fn thread_local_test_with_global_scope_var() {
    todo!();
    /*
    
      *global4_ = "abc";

      atomic_bool b(false);
      thread t([&b]() {
        EXPECT_EQ(*global4_, "");
        *global4_ = "def";
        b = true;
        EXPECT_EQ(*global4_, "def");
      });
      t.join();

      EXPECT_TRUE(b);
      EXPECT_EQ(*global4_, "abc");

    */
}

#[test] fn thread_local_test_objects_are_released() {
    todo!();
    /*
    
      static atomic<int> ctors{0};
      static atomic<int> dtors{0};
      struct A {
        A() : i() {
          ++ctors;
        }

        ~A() {
          ++dtors;
        }

        A(const A&) = delete;
        A& operator=(const A&) = delete;

        int i;
      };

      C10_DEFINE_TLS_static(A, a);

      atomic_bool b(false);
      thread t([&b]() {
        EXPECT_EQ(a->i, 0);
        a->i = 1;
        EXPECT_EQ(a->i, 1);
        b = true;
      });
      t.join();

      EXPECT_TRUE(b);

      EXPECT_EQ(ctors, 1);
      EXPECT_EQ(dtors, 1);

    */
}

#[test] fn thread_local_test_objects_are_released_by_nonstatic() {
    todo!();
    /*
    
      static atomic<int> ctors(0);
      static atomic<int> dtors(0);
      struct A {
        A() : i() {
          ++ctors;
        }

        ~A() {
          ++dtors;
        }

        A(const A&) = delete;
        A& operator=(const A&) = delete;

        int i;
      };

      atomic_bool b(false);
      thread t([&b]() {
    #if defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
        ::ThreadLocal<A> a;
    #else // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
        ::ThreadLocal<A> a([]() {
          static thread_local A var;
          return &var;
        });
    #endif // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

        EXPECT_EQ(a->i, 0);
        a->i = 1;
        EXPECT_EQ(a->i, 1);
        b = true;
      });
      t.join();

      EXPECT_TRUE(b);

      EXPECT_EQ(ctors, 1);
      EXPECT_EQ(dtors, 1);

    */
}
