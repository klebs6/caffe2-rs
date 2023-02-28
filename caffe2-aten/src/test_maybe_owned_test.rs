crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/MaybeOwned_test.cpp]

pub struct MyString {
    base: IntrusivePtrTarget,
    base2: String,
}

pub struct MaybeOwnedTest<T> {
    base:        Test,
    borrow_from: T,
    own_copy:    T,
    own_copy2:   T,
    borrowed:    MaybeOwned<T>,
    owned:       MaybeOwned<T>,
    owned2:      MaybeOwned<T>,
}

impl MaybeOwnedTest<T> {
    
    pub fn tear_down(&mut self)  {
        
        todo!();
        /*
        // Release everything to try to trigger ASAN violations in the
        // test that broke things.
        borrowFrom = T();
        ownCopy = T();
        ownCopy2 = T();

        borrowed = MaybeOwned<T>();
        owned = MaybeOwned<T>();
        owned2 = MaybeOwned<T>();
        */
    }
}

////////////// Helper implementations for intrusive_ptr. //////////////
pub fn get_sample_value_a() -> IntrusivePtr<MyString> {
    
    todo!();
        /*
            return make_intrusive<MyString>("hello");
        */
}

pub fn get_sample_value2_a() -> IntrusivePtr<MyString> {
    
    todo!();
        /*
            return make_intrusive<MyString>("goodbye");
        */
}

pub fn equal_a(
        lhs: &IntrusivePtr<MyString>,
        rhs: &IntrusivePtr<MyString>) -> bool {
    
    todo!();
        /*
            if (!lhs || !rhs) {
        return !lhs && !rhs;
      }
      return *lhs == *rhs;
        */
}

pub fn assert_borrow_a(
        mo:            &MaybeOwned<IntrusivePtr<MyString>>,
        borrowed_from: &IntrusivePtr<MyString>)  {
    
    todo!();
        /*
            EXPECT_EQ(*mo, borrowedFrom);
      EXPECT_EQ(mo->get(), borrowedFrom.get());
      EXPECT_EQ(borrowedFrom.use_count(), 1);
        */
}

pub fn assert_own_a(
        mo:        &MaybeOwned<IntrusivePtr<MyString>>,
        original:  &IntrusivePtr<MyString>,
        use_count: usize)  {

    let use_count: usize = use_count.unwrap_or(2);
    
    todo!();
        /*
            EXPECT_EQ(*mo, original);
      EXPECT_EQ(mo->get(), original.get());
      EXPECT_NE(&*mo, &original);
      EXPECT_EQ(original.use_count(), useCount);
        */
}

/////////////// Helper implementations for Tensor. ///////////////

pub fn get_sample_value_b() -> Tensor {
    
    todo!();
        /*
            return native::zeros({2, 2}).to(kCPU);
        */
}

pub fn get_sample_value2_b() -> Tensor {
    
    todo!();
        /*
            return native::ones({2, 2}).to(kCPU);
        */
}

pub fn equal_b(
    lhs: &Tensor,
    rhs: &Tensor) -> bool {

    todo!();
        /*
            if (!lhs.defined() || !rhs.defined()) {
        return !lhs.defined() && !rhs.defined();
      }
      return native::cpu_equal(lhs, rhs);
        */
}

pub fn assert_borrow_b(
    mo:            &MaybeOwned<Tensor>,
    borrowed_from: &Tensor)  {

    todo!();
        /*
            EXPECT_TRUE(mo->is_same(borrowedFrom));
      EXPECT_EQ(borrowedFrom.use_count(), 1);
        */
}

pub fn assert_own_b(
        mo:        &MaybeOwned<Tensor>,
        original:  &Tensor,
        use_count: usize)  {

    let use_count: usize = use_count.unwrap_or(2);
    
    todo!();
        /*
            EXPECT_TRUE(mo->is_same(original));
      EXPECT_EQ(original.use_count(), useCount);
        */
}

impl<T> MaybeOwnedTest<T> {
    
    pub fn set_up(&mut self)  {
        
        todo!();
        /*
            borrowFrom = getSampleValue<T>();
      ownCopy = getSampleValue<T>();
      ownCopy2 = getSampleValue<T>();
      borrowed = MaybeOwned<T>::borrowed(borrowFrom);
      owned = MaybeOwned<T>::owned(in_place, ownCopy);
      owned2 = MaybeOwned<T>::owned(T(ownCopy2));
        */
    }
}

pub type MaybeOwnedTypes = Types<IntrusivePtr<MyString>,Tensor>;

typed_test_case!{MaybeOwnedTest, MaybeOwnedTypes}

#[test] fn maybe_owned_test_simple_dereferencing_string() {
    todo!();
    /*
    
      assertBorrow(this->borrowed, this->borrowFrom);
      assertOwn(this->owned, this->ownCopy);
      assertOwn(this->owned2, this->ownCopy2);

    */
}

#[test] fn maybe_owned_test_default_ctor() {
    todo!();
    /*
    
      MaybeOwned<TypeParam> borrowed, owned;
      // Don't leave the fixture versions around messing up reference counts.
      this->borrowed = MaybeOwned<TypeParam>();
      this->owned = MaybeOwned<TypeParam>();
      borrowed = MaybeOwned<TypeParam>::borrowed(this->borrowFrom);
      owned = MaybeOwned<TypeParam>::owned(in_place, this->ownCopy);

      assertBorrow(borrowed, this->borrowFrom);
      assertOwn(owned, this->ownCopy);

    */
}

#[test] fn maybe_owned_test_copy_constructor() {
    todo!();
    /*
    

      auto copiedBorrowed(this->borrowed);
      auto copiedOwned(this->owned);
      auto copiedOwned2(this->owned2);

      assertBorrow(this->borrowed, this->borrowFrom);
      assertBorrow(copiedBorrowed, this->borrowFrom);

      assertOwn(this->owned, this->ownCopy, 3);
      assertOwn(copiedOwned, this->ownCopy, 3);
      assertOwn(this->owned2, this->ownCopy2, 3);
      assertOwn(copiedOwned2, this->ownCopy2, 3);

    */
}

#[test] fn maybe_owned_test_move_dereferencing() {
    todo!();
    /*
    
      // Need a different value.
      this->owned = MaybeOwned<TypeParam>::owned(in_place, getSampleValue2<TypeParam>());

      EXPECT_TRUE(equal(*move(this->borrowed), getSampleValue<TypeParam>()));
      EXPECT_TRUE(equal(*move(this->owned), getSampleValue2<TypeParam>()));

      // Borrowed is unaffected.
      assertBorrow(this->borrowed, this->borrowFrom);

      // Owned is a null intrusive_ptr / empty Tensor.
      EXPECT_TRUE(equal(*this->owned, TypeParam()));

    */
}

#[test] fn maybe_owned_test_move_constructor() {
    todo!();
    /*
    
      auto movedBorrowed(move(this->borrowed));
      auto movedOwned(move(this->owned));
      auto movedOwned2(move(this->owned2));

      assertBorrow(movedBorrowed, this->borrowFrom);
      assertOwn(movedOwned, this->ownCopy);
      assertOwn(movedOwned2, this->ownCopy2);

    */
}

#[test] fn maybe_owned_test_copy_assignment_into() {
    todo!();
    /*
    
      auto copiedBorrowed = MaybeOwned<TypeParam>::owned(in_place);
      auto copiedOwned = MaybeOwned<TypeParam>::owned(in_place);
      auto copiedOwned2 = MaybeOwned<TypeParam>::owned(in_place);

      copiedBorrowed = this->borrowed;
      copiedOwned = this->owned;
      copiedOwned2 = this->owned2;

      assertBorrow(this->borrowed, this->borrowFrom);
      assertBorrow(copiedBorrowed, this->borrowFrom);
      assertOwn(this->owned, this->ownCopy, 3);
      assertOwn(copiedOwned, this->ownCopy, 3);
      assertOwn(this->owned2, this->ownCopy2, 3);
      assertOwn(copiedOwned2, this->ownCopy2, 3);

    */
}

#[test] fn maybe_owned_test_copy_assignment_into_borrowed() {
    todo!();
    /*
    
      auto otherBorrowFrom = getSampleValue2<TypeParam>();
      auto otherOwnCopy = getSampleValue2<TypeParam>();
      auto copiedBorrowed = MaybeOwned<TypeParam>::borrowed(otherBorrowFrom);
      auto copiedOwned = MaybeOwned<TypeParam>::borrowed(otherOwnCopy);
      auto copiedOwned2 = MaybeOwned<TypeParam>::borrowed(otherOwnCopy);

      copiedBorrowed = this->borrowed;
      copiedOwned = this->owned;
      copiedOwned2 = this->owned2;

      assertBorrow(this->borrowed, this->borrowFrom);
      assertBorrow(copiedBorrowed, this->borrowFrom);

      assertOwn(this->owned, this->ownCopy, 3);
      assertOwn(this->owned2, this->ownCopy2, 3);
      assertOwn(copiedOwned, this->ownCopy, 3);
      assertOwn(copiedOwned2, this->ownCopy2, 3);

    */
}

#[test] fn maybe_owned_test_move_assignment_into() {
    todo!();
    /*
      auto movedBorrowed = MaybeOwned<TypeParam>::owned(in_place);
      auto movedOwned = MaybeOwned<TypeParam>::owned(in_place);
      auto movedOwned2 = MaybeOwned<TypeParam>::owned(in_place);

      movedBorrowed = move(this->borrowed);
      movedOwned = move(this->owned);
      movedOwned2 = move(this->owned2);

      assertBorrow(movedBorrowed, this->borrowFrom);
      assertOwn(movedOwned, this->ownCopy);
      assertOwn(movedOwned2, this->ownCopy2);

    */
}

#[test] fn maybe_owned_test_move_assignment_into_borrowed() {
    todo!();
    /*
      auto y = getSampleValue2<TypeParam>();
      auto movedBorrowed = MaybeOwned<TypeParam>::borrowed(y);
      auto movedOwned = MaybeOwned<TypeParam>::borrowed(y);
      auto movedOwned2 = MaybeOwned<TypeParam>::borrowed(y);

      movedBorrowed = move(this->borrowed);
      movedOwned = move(this->owned);
      movedOwned2 = move(this->owned2);

      assertBorrow(movedBorrowed, this->borrowFrom);
      assertOwn(movedOwned, this->ownCopy);
      assertOwn(movedOwned2, this->ownCopy2);

    */
}

#[test] fn maybe_owned_test_self_assignment() {
    todo!();
    /*
    
      this->borrowed = this->borrowed;
      this->owned = this->owned;
      this->owned2 = this->owned2;

      assertBorrow(this->borrowed, this->borrowFrom);
      assertOwn(this->owned, this->ownCopy);
      assertOwn(this->owned2, this->ownCopy2);

    */
}
