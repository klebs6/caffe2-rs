crate::ix!();

#[macro_export] macro_rules! delegate_packv_function {
    ($t:ty, $OriginalFunc:ident) => {
        /*
        template <>                                                   
            void PackV<T>(const int N, const T* a, const int* ia, T* y) { 
                OriginalFunc(N, a, ia, y);                                  
        }
        */
    }
}

delegate_packv_function!{f32, vsPackV}
delegate_packv_function!{f64, vdPackV}

#[macro_export] macro_rules! delegate_unpackv_function {
    ($t:ty, $OriginalFunc:ident) => {
        /*
        template <>                                                     
            void UnpackV<T>(const int N, const T* a, T* y, const int* iy) { 
                OriginalFunc(N, a, y, iy);                                    
        }
        */
    }
}

delegate_unpackv_function!{f32, vsUnpackV}
delegate_unpackv_function!{f64, vdUnpackV}

#[macro_export] macro_rules! delegate_simple_binary_function {
    ($t:ty, $Funcname:ident, $OriginalFunc:ident) => {
        /*
        template <>                                                      
            void Funcname<T>(const int N, const T* a, const T* b, T* y) {    
                OriginalFunc(N, a, b, y);                                      
        }
        */
    }
}

delegate_simple_binary_function!{f32, Pow, vsPow}
delegate_simple_binary_function!{f64, Pow, vdPow}
