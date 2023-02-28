crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/thread_name.h]
//-------------------------------------------[.cpp/pytorch/c10/util/thread_name.cpp]

#[cfg(not(__GLIBC_PREREQ))]
macro_rules! __GLIBC_PREREQ {
    ($x:ident, $y:ident) => {
        /*
                0
        */
    }
}

lazy_static!{
    /*
    #if __GLIBC__ && __GLIBC_PREREQ(2, 12) && !defined(__APPLE__) && \
        !defined(__ANDROID__)
    #define C10_HAS_PTHREAD_SETNAME_NP
    #endif
    */
}

pub fn set_thread_name(name: String)  {
    
    todo!();
        /*
            #ifdef C10_HAS_PTHREAD_SETNAME_NP
      constexpr size_t kMaxThreadName = 15;
      name.resize(min(name.size(), kMaxThreadName));

      pthread_setname_np(pthread_self(), name.c_str());
    #endif
        */
}
