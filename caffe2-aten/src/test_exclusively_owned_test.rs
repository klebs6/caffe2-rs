crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/ExclusivelyOwned_test.cpp]

pub struct MyString {
    base: IntrusivePtrTarget,
    base2: String,
}

pub struct ExclusivelyOwnedTest<T> {
    base:                Test,
    default_constructed: ExclusivelyOwned<T>,
    sample:              ExclusivelyOwned<T>,
}

impl ExclusivelyOwnedTest<T> {

    pub fn set_up(&mut self)  {
        todo!();
    }

    pub fn tear_down(&mut self)  {
        
        todo!();
        /*
            defaultConstructed = ExclusivelyOwned<T>();
        sample = ExclusivelyOwned<T>();
        */
    }
}

pub fn get_sample_value_a<T>() -> T {
    
    todo!();
        /*
        
        */
}

pub fn get_sample_value_b() -> IntrusivePtr<MyString> {
    
    todo!();
        /*
            return make_intrusive<MyString>("hello");
        */
}

pub fn get_sample_value_c() -> Tensor {
    
    todo!();
        /*
            return native::zeros({2, 2}).to(kCPU);
        */
}

pub fn assert_is_sample_object<T>(eo: &T)  {
    
    todo!();
        /*
        
        */
}

pub fn assert_is_sample_object_my_string(s: &MyString)  {
    
    todo!();
        /*
            EXPECT_STREQ(s.c_str(), "hello");
        */
}

pub fn assert_is_sample_object_intrusive_ptr_my_string(s: &IntrusivePtr<MyString>)  {
    
    todo!();
        /*
            assertIsSampleObject(*s);
        */
}

pub fn assert_is_sample_object_tensor(t: &Tensor)  {
    
    todo!();
        /*
            EXPECT_EQ(t.sizes(), (IntArrayRef{2, 2}));
      EXPECT_EQ(t.strides(), (IntArrayRef{2, 1}));
      ASSERT_EQ(t.scalar_type(), ScalarType::Float);
      static const float zeros[4] = {0};
      EXPECT_EQ(memcmp(zeros, t.data_ptr(), 4 * sizeof(float)), 0);
        */
}

impl<T> ExclusivelyOwnedTest<T> {
    
    pub fn set_up(&mut self)  {
        
        todo!();
        /*
            defaultConstructed = ExclusivelyOwned<T>();
      sample = ExclusivelyOwned<T>(getSampleValue<T>());
        */
    }
}

pub type ExclusivelyOwnedTypes = Types<IntrusivePtr<MyString>,Tensor>;

typed_test_case!{ExclusivelyOwnedTest, ExclusivelyOwnedTypes}

#[test] fn exclusively_owned_test_default_constructor() {
    todo!();
    /*
    
      ExclusivelyOwned<TypeParam> defaultConstructed;

    */
}

#[test] fn exclusively_owned_test_move_constructor() {
    todo!();
    /*
    
      auto movedDefault = move(this->defaultConstructed);
      auto movedSample = move(this->sample);

      assertIsSampleObject(*movedSample);

    */
}

#[test] fn exclusively_owned_test_move_assignment() {
    todo!();
    /*
    
      // Move assignment from a default-constructed ExclusivelyOwned is handled in
      // TearDown at the end of every test!
      ExclusivelyOwned<TypeParam> anotherSample = ExclusivelyOwned<TypeParam>(getSampleValue<TypeParam>());
      anotherSample = move(this->sample);
      assertIsSampleObject(*anotherSample);

    */
}

#[test] fn exclusively_owned_test_move_assignment_from_contained_type() {
    todo!();
    /*
    
      ExclusivelyOwned<TypeParam> anotherSample = ExclusivelyOwned<TypeParam>(getSampleValue<TypeParam>());
      anotherSample = getSampleValue<TypeParam>();
      assertIsSampleObject(*anotherSample);

    */
}

#[test] fn exclusively_owned_test_take() {
    todo!();
    /*
    
      auto x = move(this->sample).take();
      assertIsSampleObject(x);

    */
}

pub fn inspect_tensor()  {
    
    todo!();
        /*
            auto t = getSampleValue<Tensor>();
        */
}

pub fn inspect_exclusively_owned_tensor()  {
    
    todo!();
        /*
            ExclusivelyOwned<Tensor> t(getSampleValue<Tensor>());
        */
}

pub fn inspect_intrusive_ptr()  {
    
    todo!();
        /*
            auto p = getSampleValue<intrusive_ptr<MyString>>();
        */
}

pub fn inspect_exclusively_owned_intrusive_ptr()  {
    
    todo!();
        /*
            ExclusivelyOwned<intrusive_ptr<MyString>> p(getSampleValue<intrusive_ptr<MyString>>());
        */
}

pub fn inspect_unique_ptr()  {
    
    todo!();
        /*
            unique_ptr<MyString> p(getSampleValue<intrusive_ptr<MyString>>().release());
        */
}

