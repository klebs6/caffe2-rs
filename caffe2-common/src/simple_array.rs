crate::ix!();

pub struct SimpleArray<T, const N: usize> {
    data: [T; N],
}

pub const kCUDATensorMaxDims: usize = 8;
