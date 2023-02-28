/// generic tests

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/util/ConstexprCrc_test.cpp]

/// These used to be const_assert, and still should be
///
#[test] fn test_crc64() {

    // crc64 is deterministic"
    assert_eq!{
        Crc64::from_str("MyTestString"),
        Crc64::from_str("MyTestString")
    }

    // different strings, different result
    assert_ne!{
        Crc64::from_str("MyTestString1"), 
        Crc64::from_str("MyTestString2")
    }

    /**
      | check concrete expected values (for
      | CRC64 with
      | 
      | Jones coefficients and an init value
      | of 0)
      |
      */
    lazy_static!{
        /*
        const_assert!{crc64_t{0} == crc64("")}
        const_assert!{crc64_t{0xe9c6d914c4b8d9ca} == crc64("123456789")}
        */
    }

}
