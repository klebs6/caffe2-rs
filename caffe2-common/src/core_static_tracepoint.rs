crate::ix!();

#[cfg(all(__elf__, any(__x86_64__, __i386__)))]
#[macro_export] macro_rules! caffe_sdt {
    ($name:ident, $($arg:ident),*) => {
        todo!();
        /*
        
          CAFFE_SDT_PROBE_N(                                                 
            caffe2, name, CAFFE_SDT_NARG(0, ##__VA_ARGS__), ##__VA_ARGS__)
        */
    }
}

#[cfg(not(all(__elf__, any(__x86_64__, __i386__))))]
#[macro_export] macro_rules! caffe_sdt {
    ($name:ident, $($arg:ident),*) => {
        todo!();
        /*
                do {} while(0)
        */
    }
}
