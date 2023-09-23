crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/variant_test.cpp]

// NOTE: We need to provide the default
// constructor for each struct, otherwise Clang
// 3.8 would complain:
//
// ```
// error: default initialization of an object of const type 'const enumtype::Enum1'
// without a user-provided default constructor
// ```
pub struct Enum1 { }
pub struct Enum2 { }
pub struct Enum3 { }

pub struct EnumName {

}

impl EnumName {
    
    pub fn invoke(&self, v: &mut Enum1) -> String {
        
        todo!();
        /*
            return "Enum1";
        */
    }
    
    pub fn invoke(&self, v: &mut Enum2) -> String {
        
        todo!();
        /*
            return "Enum2";
        */
    }
    
    pub fn invoke(&self, v: &mut Enum3) -> String {
        
        todo!();
        /*
            return "Enum3";
        */
    }
}

lazy_static!{
    /*
    const Enum1 kEnum1;
    const Enum2 kEnum2;
    const Enum3 kEnum3;
    */
}

pub fn func(v: Variant<Enum1,Enum2,Enum3>) -> String {
    
    todo!();
        /*
            if (get_if<testns::enumtype::Enum1>(&v)) {
        return "Enum1";
      } else if (get_if<testns::enumtype::Enum2>(&v)) {
        return "Enum2";
      } else if (get_if<testns::enumtype::Enum3>(&v)) {
        return "Enum3";
      } else {
        return "Unsupported enum";
      }
        */
}

#[test] fn variant_test_basic() {
    todo!();
    /*
    
      ASSERT_EQ(func(testns::kEnum1), "Enum1");
      ASSERT_EQ(func(testns::kEnum2), "Enum2");
      ASSERT_EQ(func(testns::kEnum3), "Enum3");

      variant<testns::enumtype::Enum1, testns::enumtype::Enum2, testns::enumtype::Enum3> v;
      {
        v = testns::kEnum1;
        ASSERT_EQ(visit(testns::enum_name{}, v), "Enum1");
      }
      {
        v = testns::kEnum2;
        ASSERT_EQ(visit(testns::enum_name{}, v), "Enum2");
      }
      {
        v = testns::kEnum3;
        ASSERT_EQ(visit(testns::enum_name{}, v), "Enum3");
      }

    */
}
