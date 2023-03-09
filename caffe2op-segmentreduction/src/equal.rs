crate::ix!();

/**
  | Helper function to enforce naming conventions
  | at compile time.
  |
  */
#[inline] pub fn equal(
    lhs:  *const u8,
    rhs1: *const u8,
    rhs2: *const u8,
    rhs3: *const u8) -> bool {
    
    todo!();
    /*
        return (*lhs == 0 && *rhs1 == 0 && *rhs2 == 0 && *rhs3 == 0) ||
          (*rhs1 != 0 && *lhs == *rhs1 && equal(lhs + 1, rhs1 + 1, rhs2, rhs3)) ||
          (*rhs1 == 0 && *rhs2 != 0 && *lhs == *rhs2 &&
           equal(lhs + 1, rhs1, rhs2 + 1, rhs3)) ||
          (*rhs1 == 0 && *rhs2 == 0 && *rhs3 != 0 && *lhs == *rhs3 &&
           equal(lhs + 1, rhs1, rhs2, rhs3 + 1));
    */
}
