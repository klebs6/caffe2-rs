crate::ix!();

#[inline] pub fn sigmoid<T: Float>(x: T) -> T where f64: From<T> {
    let x = f64::from(x);
    T::from(1.0 / (1.0 + (-x).exp())).unwrap()
}

#[inline] pub fn host_tanh<T: Float>(x: T) -> T where f64: From<T> {
    T::from(2.0 * f64::from(sigmoid(T::from(2.0 * f64::from(x)).unwrap())) - 1.0).unwrap()
}
