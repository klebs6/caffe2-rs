#[macro_export] macro_rules! x { 
    ($x:ident) => { 
        mod $x; 
        pub use $x::*; 
    }
}

#[macro_export] macro_rules! ix { 
    () => { 
        use crate::{ 
            imports::* , 
        };
        use crate::*;
    } 
}

/// This is a temporary fix for certain types which live in aten or elsewhere
///
#[derive(Default)]
pub struct Unknown {}
pub type TypeMeta = Unknown;
pub type Blob = Unknown;

pub use rand::StdRng;
pub use statrs::distribution::Uniform;

#[macro_use]
extern crate derive_error;

pub use std::error::Error;
pub use derive_error::*;
pub use std::iter::Iterator;
pub use libc;
pub use rand;
pub use ndarray;
pub use pyo3::prelude::*;
pub use aligned::*;
pub use intrusive_collections::intrusive_adapter;
pub use intrusive_collections::{LinkedList as IntrusiveLinkedList, LinkedListLink};
pub use protobuf;
pub use parking_lot;
pub use threadpool;
pub use cblas_sys;
pub use mt19937;
pub use statrs;
pub use nalgebra;
pub use std::mem::size_of;
pub use num::complex::ComplexFloat;

pub use nalgebra::{
    Complex
};

pub use ::atomic::Atomic;
pub use protobuf::RepeatedField;
pub use hex_literal::*;

pub use caffe2_proto::*;
pub use caffe2_derive::*;

use cudnn::cudnnDataType_t::{
    CUDNN_DATA_FLOAT,
    CUDNN_DATA_HALF,
};

pub use lazy_static::*;
pub use static_assertions::*;
pub use std::cell::*;
pub use std::collections::*;
pub use std::ops::*;
pub use std::rc::*;
pub use std::sync::*;
pub use core::ffi::c_void;
pub use smallvec::SmallVec;
pub use rayon::prelude::*;

pub use half::{f16,bf16};
pub use crossbeam::queue::SegQueue;

pub use byteorder::{
    NativeEndian, 
    LittleEndian, 
    BigEndian
};

pub use std::collections::{
    HashMap,
    LinkedList,
    HashSet,
};

pub use std::ops::{
    Deref,
    DerefMut,
    AddAssign,
};

pub use std::sync::{
    Condvar,
    Mutex,
    MutexGuard,
    Arc,
    atomic::{
        AtomicU32,
        AtomicPtr,
        AtomicI32,
        AtomicUsize,
        AtomicBool,
    },
};

pub use lazy_static::lazy_static;
pub use std::fmt;
pub use num_traits::int::PrimInt;
pub use num_traits::real::Real;
pub use std::marker::PhantomData;
pub use std::hash::Hash;
pub use std::hash::Hasher;
pub use paste::paste;
pub use core::ptr::NonNull;

pub use structopt::*;
pub use structopt_derive::*;
pub use pod::Pod;

/*
pub use miopen_sys::{
    miopenDataType_t,
    miopenDataType_t::miopenHalf,
    miopenHandle_t,
    miopenStatus_t,
    miopenTensorDescriptor_t,
    miopenRNNDescriptor_t,
    miopenActivationDescriptor_t,
};
*/
pub type miopenDataType_t             = Unknown;
pub type miopenHalf                   = Unknown;
pub type miopenHandle_t               = Unknown;
pub type miopenStatus_t               = Unknown;
pub type miopenTensorDescriptor_t     = Unknown;
pub type miopenRNNDescriptor_t        = Unknown;
pub type miopenActivationDescriptor_t = Unknown;

pub use opencv::core as cv;
pub use core::cmp::Ordering;
pub use zmq_sys::zmq_msg_t;

/*
pub use hip_sys::hipblas::{
    hipEvent_t,
    hipStream_t,
    hipCtx_t,
};
*/
pub type hipEvent_t = Unknown;
pub type hipStream_t = Unknown;
pub type hipCtx_t = Unknown;

pub use bitflags::*;

#[cfg(target_arch = "arm")]
pub use core::arch::arm::{
    float32x4_t,
    uint8x8x4_t,
};

pub use test_context::TestContext;
pub use std::io::{BufReader,BufWriter};
pub use std::io::{Read,Write};
pub use core::ops::Index;
pub use num::{Num,Float};
pub use enhanced_enum::enhanced_enum;
//pub use ndarray::ArrayBase;
pub use std::thread::Thread;
pub use parking_lot::RawMutex;

pub(crate) use stdext::compile_warning;

#[macro_export] macro_rules! ternary {
    ($condition:expr,$if_true:expr,$if_false:expr) => {
        match $condition {
            true => $if_true,
            false => $if_false,
        }
    }
}
