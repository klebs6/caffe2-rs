crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/qint32.h]

pub type Align4<T> = Aligned<A4,T>;

/**
  | qint32 is for signed 32 bit quantized
  | Tensors
  |
  */
#[derive(Default)]
pub struct qint32 {
    val: Align4<i32>,
}

pub type underlying = i32;

impl qint32 {
    
    pub fn new(val: i32) -> Self {
    
        todo!();
        /*
        : val(val),

        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/util/qint8.h]

pub type Qint8Underlying = i8;

/**
  | This is the data type for quantized Tensors.
  | Right now we only have qint8 which is
  | for 8 bit Tensors, and qint32 for 32 bit
  | int Tensors, we might have 4 bit, 2 bit
  | or 1 bit data types in the future.
  |
  */
#[derive(Default)]
pub struct qint8 {

    val: i8,
}

impl qint8 {
    
    pub fn new(val: i8) -> Self {
    
        todo!();
        /*
        : val(val),

        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/util/quint4x2.h]

pub type Quint4x2Underlying = u8;

/**
  | quint4x2 is for un-signed 4 bit quantized
  | Tensors that are packed to byte boundary.
  |
  */
pub struct Quint4x2 {
    val: u8,
}

impl Quint4x2 {
    
    pub fn new(val: u8) -> Self {
    
        todo!();
        /*
        : val(val),
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/util/quint8.h]

pub type Quint8Underlying = u8;

/**
  | quint8 is for unsigned 8 bit quantized
  | Tensors
  |
  */
pub struct qui8 {
    val: u8,
}

impl qui8 {
    
    pub fn new(val: u8) -> Self {
    
        todo!();
        /*
        : val(val),
        */
    }
}
