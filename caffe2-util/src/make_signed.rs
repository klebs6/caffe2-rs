pub trait GetSigned {
    type Type;
}

#[macro_export] macro_rules! impl_signed {
    ($src:ty, $dst:ty) => { impl GetSigned for $src { type Type = $dst; } }
}

impl_signed![u8, i8];
impl_signed![i8, i8];
impl_signed![u16,i16];
impl_signed![i16,i16];
impl_signed![u32,i32];
impl_signed![i32,i32];
impl_signed![u64,i64];
impl_signed![i64,i64];

