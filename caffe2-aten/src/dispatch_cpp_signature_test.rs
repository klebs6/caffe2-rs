crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/dispatch/CppSignature_test.cpp]

#[test] fn cpp_signature_test_given_equal_signature_then_are_equal() {
    todo!();
    /*
    
        EXPECT_EQ(CppSignature::make<void()>(), CppSignature::make<void()>());
        EXPECT_EQ(CppSignature::make<i64(std::string, i64)>(), CppSignature::make<i64(std::string, i64)>());

    */
}

#[test] fn cpp_signature_test_given_different_signature_then_are_different() {
    todo!();
    /*
    
        EXPECT_NE(CppSignature::make<void()>(), CppSignature::make<i64()>());
        EXPECT_NE(CppSignature::make<i64(std::string)>(), CppSignature::make<i64(std::string, i64)>());
        EXPECT_NE(CppSignature::make<std::string(std::string)>(), CppSignature::make<i64(std::string)>());

    */
}

#[test] fn cpp_signature_test_given_equal_functor_and_function_then_are_equal() {
    todo!();
    /*
    
        struct Functor final {
            i64 operator()(std::string) {return 0;}
        };
        EXPECT_EQ(CppSignature::make<Functor>(), CppSignature::make<i64(std::string)>());

    */
}

#[test] fn cpp_signature_test_given_different_functor_and_function_then_are_different() {
    todo!();
    /*
    
        struct Functor final {
            i64 operator()(std::string) {return 0;}
        };
        EXPECT_NE(CppSignature::make<Functor>(), CppSignature::make<i64(std::string, i64)>());

    */
}

#[test] fn cpp_signature_test_given_then_can_query_name_without_crashing() {
    todo!();
    /*
    
        CppSignature::make<void(i64, const i64&)>().name();

    */
}
