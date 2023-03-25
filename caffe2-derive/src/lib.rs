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
pub fn USE_CPU_CONTEXT_OPERATOR_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn NOMNIGRAPH_DEFINE_NN_RTTI(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_IDEEP_DEF_ALIASES(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_IDEEP_OPERATOR_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_IDEEP_CONV_POOL_BASE_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_IDEEP_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
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

#[proc_macro_attribute]
pub fn USE_CONV_POOL_BASE_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_DISPATCH_HELPER(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_SIMPLE_CTOR_DTOR(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_RECURRENT_BASE_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS(attr: TokenStream, item: TokenStream) -> TokenStream {
    macro_rules! USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS {
        ($Context:ident) => {
            todo!();
            /*
            USE_OPERATOR_FUNCTIONS(Context);                        
            using ConvTransposeUnpoolBase<Context>::kernel_;        
            using ConvTransposeUnpoolBase<Context>::kernel_h;       
            using ConvTransposeUnpoolBase<Context>::kernel_w;       
            using ConvTransposeUnpoolBase<Context>::stride_;        
            using ConvTransposeUnpoolBase<Context>::stride_h;       
            using ConvTransposeUnpoolBase<Context>::stride_w;       
            using ConvTransposeUnpoolBase<Context>::pads_;          
            using ConvTransposeUnpoolBase<Context>::pad_t;          
            using ConvTransposeUnpoolBase<Context>::pad_l;          
            using ConvTransposeUnpoolBase<Context>::pad_b;          
            using ConvTransposeUnpoolBase<Context>::pad_r;          
            using ConvTransposeUnpoolBase<Context>::adj_;           
            using ConvTransposeUnpoolBase<Context>::group_;         
            using ConvTransposeUnpoolBase<Context>::order_;         
            using ConvTransposeUnpoolBase<Context>::shared_buffer_; 
            using ConvTransposeUnpoolBase<Context>::ws_
            */
        }
    }
    item
}
