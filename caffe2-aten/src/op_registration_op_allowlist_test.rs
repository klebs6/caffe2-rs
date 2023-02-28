crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/op_registration/op_allowlist_test.cpp]

pub mod op_allowlist_contains_test {

    use super::*;

    const_assert!{op_allowlist_contains("", "")}

    const_assert!{!op_allowlist_contains("", "a")}
    const_assert!{!op_allowlist_contains("a", "")}
    const_assert!{!op_allowlist_contains("a;bc", "")}

    const_assert!{op_allowlist_contains("a;bc;d", "a")}
    const_assert!{op_allowlist_contains("a;bc;d", "bc")}
    const_assert!{op_allowlist_contains("a;bc;d", "d")}
    const_assert!{!op_allowlist_contains("a;bc;d", "e")}
    const_assert!{!op_allowlist_contains("a;bc;d", "")}

    const_assert!{op_allowlist_contains(";", "")}
    const_assert!{op_allowlist_contains("a;", "")}
    const_assert!{op_allowlist_contains("a;", "a")}
}
