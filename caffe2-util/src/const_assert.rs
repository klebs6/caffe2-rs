crate::ix!();

/// Usage: `const_assert!(Var1: Ty, Var2: Ty, ... => expression)`
#[macro_export] macro_rules! const_assertx {
    ($($list:ident : $ty:ty),* => $expr:expr) => {{
        struct Assert<$(const $list: usize,)*>;
        impl<$(const $list: $ty,)*> Assert<$($list,)*> {
            const OK: u8 = 0 - !($expr) as u8;
        }
        Assert::<$($list,)*>::OK
    }};
    ($expr:expr) => {
        const OK: u8 = 0 - !($expr) as u8;
    };
}

fn gt<const X: usize, const Y: usize>() {
    const_assertx!(X: usize, Y: usize => X > Y);
}

const fn is_prime(n: usize) -> bool {
    let mut i = 2;
    while i*i <= n {
        if n % i == 0 {
            return false;
        }
        i += 1;
    }
    return true;
}

fn prime<const N: usize>() {
    const_assertx!(N: usize => is_prime(N));
}


