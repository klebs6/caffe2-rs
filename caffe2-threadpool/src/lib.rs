#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{util_threadpool_pthreadpool_impl}
x!{util_threadpool_pthreadpool}
x!{util_threadpool_thread_pool_guard}
x!{util_threadpool_threadpool}
x!{util_threadpool_workerspool}
