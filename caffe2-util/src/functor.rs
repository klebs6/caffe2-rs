
pub trait Invoke {
    fn invoke();
}

pub struct NoOpFunctor {

}

impl Invoke for NoOpFunctor {
    fn invoke() {}
}
