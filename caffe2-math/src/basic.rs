crate::ix!();

/**
  | Function uses casting from int to unsigned
  | to compare if value of parameter a is
  | greater or equal to zero and lower than
  | value of parameter b.
  | 
  | The b parameter is of type signed and
  | is always positive,
  | 
  | therefore its value is always lower
  | than 0x800... where casting negative
  | value of a parameter converts it to value
  | higher than 0x800...
  | 
  | The casting allows to use one condition
  | instead of two.
  |
  */
#[inline] pub fn is_age_zero_and_altB(a: i32, b: i32) -> bool {

    todo!();
    /*
        return static_cast<unsigned int>(a) < static_cast<unsigned int>(b);
    */
}

/**
  | Calculates ceil(a / b). User must be
  | careful to ensure that there is no overflow
  | or underflow in the calculation.
  |
  */
#[inline] pub fn div_up<T>(a: T, b: T) -> T {
    todo!();
    /*
        return (a + b - T(1)) / b;
    */
}

/**
  | Rounds a up to the next highest multiple of
  | b. User must be careful to ensure that there
  | is no overflow or underflow in the calculation
  | of divUp.
  */
#[inline] pub fn round_up<T>(a: T, b: T) -> T {
    todo!();
    /*
        return DivUp<T>(a, b) * b;
    */
}

/// Returns log2(n) for a positive integer type
#[inline] pub fn integer_log2<T>(n: T, p: i32) -> i32 {
    todo!();
    /*
        return (n <= 1) ? p : IntegerLog2(n / 2, p + 1);
    */
}

/**
  | Returns the next highest power-of-2
  | for an integer type
  |
  */
#[inline] pub fn integer_next_highest_power_of2<T>(v: T) -> T {
    todo!();
    /*
        return (IntegerIsPowerOf2(v) ? T(2) * v : (T(1) << (IntegerLog2(v) + 1)));
    */
}


#[inline] pub fn not<T>(x: T) -> T {
    todo!();
    /*
        return !x;
    */
}

#[inline] pub fn sign<T>(x: T) -> T {
    todo!();
    /*
        return x > 0 ? T(1) : (x < 0 ? T(-1) : T(0));
    */
}

#[inline] pub fn negate<T>(x: T) -> T {
    todo!();
    /*
        return -x;
    */
}

#[inline] pub fn inv<T>(x: T) -> T {
    todo!();
    /*
        return T(1) / x;
    */
}

#[inline] pub fn square<T>(x: T) -> T {
    todo!();
    /*
        return x * x;
    */
}

#[inline] pub fn cube<T>(x: T) -> T {
    todo!();
    /*
        return x * x * x;
    */
}
