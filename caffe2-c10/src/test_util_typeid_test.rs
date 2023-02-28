crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/util/typeid_test.cpp]

pub struct TypeMetaTestFoo {}
pub struct TypeMetaTestBar {}

caffe_known_type!{
    TypeMetaTestFoo
}

caffe_known_type!{
    TypeMetaTestBar
}

#[test] fn type_meta_test_static() {
    todo!();
    /*
    
      EXPECT_EQ(TypeMeta::ItemSize<int>(), sizeof(int));
      EXPECT_EQ(TypeMeta::ItemSize<float>(), sizeof(float));
      EXPECT_EQ(TypeMeta::ItemSize<TypeMetaTestFoo>(), sizeof(TypeMetaTestFoo));
      EXPECT_EQ(TypeMeta::ItemSize<TypeMetaTestBar>(), sizeof(TypeMetaTestBar));
      EXPECT_NE(TypeMeta::Id<int>(), TypeMeta::Id<float>());
      EXPECT_NE(TypeMeta::Id<int>(), TypeMeta::Id<TypeMetaTestFoo>());
      EXPECT_NE(TypeMeta::Id<TypeMetaTestFoo>(), TypeMeta::Id<TypeMetaTestBar>());
      EXPECT_EQ(TypeMeta::Id<int>(), TypeMeta::Id<int>());
      EXPECT_EQ(TypeMeta::Id<TypeMetaTestFoo>(), TypeMeta::Id<TypeMetaTestFoo>());

    */
}

#[test] fn type_meta_test_names() {
    todo!();
    /*
    
      TypeMeta null_meta;
      EXPECT_EQ("nullptr (uninitialized)", null_meta.name());
      TypeMeta int_meta = TypeMeta::Make<int>();
      EXPECT_EQ("int", int_meta.name());
      TypeMeta string_meta = TypeMeta::Make<string>();
      EXPECT_TRUE(string_view::npos != string_meta.name().find("string"));

    */
}

#[test] fn type_meta_test() {
    todo!();
    /*
    
      TypeMeta int_meta = TypeMeta::Make<int>();
      TypeMeta float_meta = TypeMeta::Make<float>();
      TypeMeta foo_meta = TypeMeta::Make<TypeMetaTestFoo>();
      TypeMeta bar_meta = TypeMeta::Make<TypeMetaTestBar>();

      TypeMeta another_int_meta = TypeMeta::Make<int>();
      TypeMeta another_foo_meta = TypeMeta::Make<TypeMetaTestFoo>();

      EXPECT_EQ(int_meta, another_int_meta);
      EXPECT_EQ(foo_meta, another_foo_meta);
      EXPECT_NE(int_meta, float_meta);
      EXPECT_NE(int_meta, foo_meta);
      EXPECT_NE(foo_meta, bar_meta);
      EXPECT_TRUE(int_meta.Match<int>());
      EXPECT_TRUE(foo_meta.Match<TypeMetaTestFoo>());
      EXPECT_FALSE(int_meta.Match<float>());
      EXPECT_FALSE(int_meta.Match<TypeMetaTestFoo>());
      EXPECT_FALSE(foo_meta.Match<int>());
      EXPECT_FALSE(foo_meta.Match<TypeMetaTestBar>());
      EXPECT_EQ(int_meta.id(), TypeMeta::Id<int>());
      EXPECT_EQ(float_meta.id(), TypeMeta::Id<float>());
      EXPECT_EQ(foo_meta.id(), TypeMeta::Id<TypeMetaTestFoo>());
      EXPECT_EQ(bar_meta.id(), TypeMeta::Id<TypeMetaTestBar>());
      EXPECT_EQ(int_meta.itemsize(), TypeMeta::ItemSize<int>());
      EXPECT_EQ(float_meta.itemsize(), TypeMeta::ItemSize<float>());
      EXPECT_EQ(foo_meta.itemsize(), TypeMeta::ItemSize<TypeMetaTestFoo>());
      EXPECT_EQ(bar_meta.itemsize(), TypeMeta::ItemSize<TypeMetaTestBar>());
      EXPECT_EQ(int_meta.name(), "int");
      EXPECT_EQ(float_meta.name(), "float");
      EXPECT_NE(foo_meta.name().find("TypeMetaTestFoo"), string_view::npos);
      EXPECT_NE(bar_meta.name().find("TypeMetaTestBar"), string_view::npos);

    */
}

pub struct ClassAllowAssignment {
    x: i32,
}

impl Default for ClassAllowAssignment {
    
    fn default() -> Self {
        todo!();
        /*
        : x(42),

        
        */
    }
}

impl ClassAllowAssignment {

    // NOLINTNEXTLINE(modernize-use-equals-default)
    pub fn new(src: &ClassAllowAssignment) -> Self {
    
        todo!();
        /*
        : x(src.x),

        
        */
    }
}

//-----------------------------
pub struct ClassNoAssignment {
    x: i32,
}

impl Default for ClassNoAssignment {
    
    fn default() -> Self {
        todo!();
        /*
        : x(42),

        
        */
    }
}

caffe_known_type!{
    ClassAllowAssignment
}

caffe_known_type!{
    ClassNoAssignment
}

#[test] fn type_meta_test_ctor_dtor_and_copy() {
    todo!();
    /*
    
      TypeMeta fundamental_meta = TypeMeta::Make<int>();
      EXPECT_EQ(fundamental_meta.placementNew(), nullptr);
      EXPECT_EQ(fundamental_meta.placementDelete(), nullptr);
      EXPECT_EQ(fundamental_meta.copy(), nullptr);

      TypeMeta meta_a = TypeMeta::Make<ClassAllowAssignment>();
      EXPECT_TRUE(meta_a.placementNew() != nullptr);
      EXPECT_TRUE(meta_a.placementDelete() != nullptr);
      EXPECT_TRUE(meta_a.copy() != nullptr);
      ClassAllowAssignment src;
      src.x = 10;
      ClassAllowAssignment dst;
      EXPECT_EQ(dst.x, 42);
      meta_a.copy()(&src, &dst, 1);
      EXPECT_EQ(dst.x, 10);

      TypeMeta meta_b = TypeMeta::Make<ClassNoAssignment>();

      EXPECT_TRUE(meta_b.placementNew() != nullptr);
      EXPECT_TRUE(meta_b.placementDelete() != nullptr);
    #ifndef __clang__
      // gtest seems to have some problem with function pointers and
      // clang right now... Disabling it.
      // TODO: figure out the real cause.
      EXPECT_EQ(meta_b.copy(), &(_CopyNotAllowed<ClassNoAssignment>));
    #endif

    */
}

#[test] fn type_meta_test_float_16is_not_uint16() {
    todo!();
    /*
    
      EXPECT_NE(TypeMeta::Id<uint16_t>(), TypeMeta::Id<f16>());

    */
}
