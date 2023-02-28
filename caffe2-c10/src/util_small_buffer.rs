/*! 
 | Helper class for allocating temporary fixed
 | size arrays with SBO.
 |
 | This is intentionally much simpler than
 | SmallVector, to improve performace at the
 |  expense of many features:
 |
 | - No zero-initialization for numeric types
 | - No resizing after construction
 | - No copy/move
 | - No non-trivial types
 */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/SmallBuffer.h]

pub struct SmallBuffer<T: Pod,const N: usize> {
    storage: [T; N],
    size:    usize,
    data:    *mut T,
}

impl<T: Pod,const N: usize> Drop for SmallBuffer<T,N> {

    fn drop(&mut self) {
        todo!();
        /*
            if (size_ > N) {
          delete[] data_;
        }
        */
    }
}

impl<T: Pod,const N: usize> SmallBuffer<T,N> {
    
    pub fn new(size: usize) -> Self {
    
        todo!();
        /*
        : size(size),

            if (size > N) {
          data_ = new T[size];
        } else {
          data_ = &storage_[0];
        }
        */
    }
}

impl<T: Pod,const N: usize> IndexMut<i64> for SmallBuffer<T,N> {
    
    #[inline] fn index_mut(&mut self, idx: i64) -> &mut Self::Output {
        todo!();
        /*
            return data()[idx];
        */
    }
}

impl<T: Pod,const N: usize> Index<i64> for SmallBuffer<T,N> {

    type Output = T;
    
    #[inline] fn index(&self, idx: i64) -> &Self::Output {
        todo!();
        /*
            return data()[idx];
        */
    }
}

impl<T: Pod,const N: usize> SmallBuffer<T,N> {
    
    pub fn data_mut(&mut self) -> *mut T {
        
        todo!();
        /*
            return data_;
        */
    }
    
    pub fn data(&self) -> *const T {
        
        todo!();
        /*
            return data_;
        */
    }
    
    pub fn size(&self) -> usize {
        
        todo!();
        /*
            return size_;
        */
    }
}

lazy_static!{
    /*
    /// Here's an example implementation of `IntoIterator` for `SmallBuffer` that yields mutable
    /// references to the buffer elements:
    ///
    impl<'a, T: Pod, const N: usize> IntoIterator for &'a mut SmallBuffer<T, N> {
        type Item = &'a mut T;
        type IntoIter = SmallBufferIterator<'a, T, N>;

        fn into_iter(self) -> Self::IntoIter {
            SmallBufferIterator {
                slice: &mut self.storage,
                pos: 0,
            }
        }
    }

    pub struct SmallBufferIterator<'a, T, const N: usize> {
        slice: &'a mut [T; N],
        pos: usize,
    }

    /// This implementation allows you to iterate over the elements of a mutable `SmallBufferMut`
    /// reference and yields mutable references to the elements. Here's an example usage:
    ///
    impl<'a, T, const N: usize> Iterator for SmallBufferIterator<'a, T, N> {
        type Item = &'a mut T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.pos < N {
                let item = &mut self.slice[self.pos];
                self.pos += 1;
                Some(item)
            } else {
                None
            }
        }
    }

    /// In this example, we create a `SmallBufferMut` with some values and then use a `for` loop to
    /// iterate over its elements and multiply each one by 2.
    ///
    #[test] fn test_small_buffer_iterator() {
        let mut buffer = SmallBufferMut::<i32, 5>::new();
        buffer[0] = 10;
        buffer[1] = 20;
        buffer[2] = 30;
        buffer[3] = 40;
        buffer[4] = 50;

        for item in &mut buffer {
            *item *= 2;
        }

        assert_eq!(buffer, SmallBufferMut { buffer: [ 20, 40, 60, 80, 100 ] });
    }
    */
}

