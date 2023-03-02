extern crate proc_macro;

use proc_macro::TokenStream;
//use quote::quote;
//use syn;

#[proc_macro_attribute]
pub fn test_context(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn __global__(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_OPERATOR_CONTEXT_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_OPERATOR_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_DEFORMABLE_CONV_BASE_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn __ubsan_ignore_undefined__(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn compile_warning(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn launch_bounds(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn no_copy(attr: TokenStream, item: TokenStream) -> TokenStream {
    /*
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast = syn::parse(input).unwrap();

    // Build the trait implementation
    impl_hello_macro(&ast)
    */
    item
}

#[proc_macro_attribute]
pub fn noreturn(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn C10_HOST_CONSTEXPR(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
