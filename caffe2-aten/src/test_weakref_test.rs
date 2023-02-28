crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/weakref_test.cpp]

/// Weak pointer tests gets invalidated
#[test] fn test_weak_pointer_gets_invalidated() {
    todo!();
    /*
    
      IValue a = ones({2, 2});
      WeakIValue b = a;
      a = IValue();
      ASSERT_TRUE(b.lock().isNone());

    */
}

/// can successfully lock
#[test] fn test_weak_pointer_lock() {
    todo!();
    /*
    
      IValue a = ones({2, 2});
      WeakIValue b = a;
      auto c = b.lock();
      ASSERT_TRUE(c.isTensor());

      a = IValue();
      ASSERT_TRUE(!b.lock().isNone());
      c = IValue();
      ASSERT_TRUE(b.lock().isNone());

    */
}

/// updates refcounts correctly
#[test] fn test_weak_pointer_updates_refcounts() {
    todo!();
    /*
    
      Tensor a = ones({2, 2});
      ASSERT_EQ(a.use_count(), 1);
      ASSERT_EQ(a.weak_use_count(), 1);
      {
        WeakIValue b = IValue(a);
        ASSERT_EQ(a.use_count(), 1);
        ASSERT_EQ(a.weak_use_count(), 2);
      }
      ASSERT_EQ(a.use_count(), 1);
      ASSERT_EQ(a.weak_use_count(), 1);
      {
        WeakIValue b = IValue(a);
        ASSERT_EQ(a.use_count(), 1);
        auto locked = b.lock();
        ASSERT_FALSE(locked.isNone());
        ASSERT_EQ(a.use_count(), 2);
      }
      ASSERT_EQ(a.use_count(), 1);
      ASSERT_EQ(a.weak_use_count(), 1);
      {
        WeakIValue b = IValue(a);
        ASSERT_EQ(a.use_count(), 1);
        ASSERT_EQ(a.weak_use_count(), 2);
        a.reset();
        ASSERT_EQ(b.use_count(), 0);
        ASSERT_EQ(b.weak_use_count(), 1);
      }

    */
}
