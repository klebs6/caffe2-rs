crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorOperators.h]

impl Tensor {
    
    #[inline] pub fn assign_from(&mut self, rhs: &Tensor) -> &mut Tensor {
        
        todo!();
        /*
            return copy_(rhs);
        */
    }
    
    #[inline] pub fn assign_from(&mut self, rhs: Tensor) -> &mut Tensor {
        
        todo!();
        /*
            return copy_(rhs);
        */
    }
    
    #[inline] pub fn assign_from(&mut self, v: Scalar) -> &mut Tensor {
        
        todo!();
        /*
            return fill_(v);
        */
    }
    
    #[inline] pub fn operator_tilde(&self) -> Tensor {
        
        todo!();
        /*
            return bitwise_not();
        */
    }
}

impl Neg for Tensor {

    type Output = Self;
    
    fn neg(self) -> Self::Output {
        todo!();
        /*
            return neg();
        */
    }
}

impl AddAssign<&Tensor> for Tensor {
    
    fn add_assign(&mut self, other: &&Tensor) -> Self::Output {
        todo!();
        /*
            return add_(other);
        */
    }
}

impl AddAssign<Scalar> for Tensor {
    
    fn add_assign(&mut self, other: &Scalar) -> Self::Output {
        todo!();
        /*
            return add_(other);
        */
    }
}

impl SubAssign<&Tensor> for Tensor {
    
    fn sub_assign(&mut self, other: &&Tensor) -> Self::Output {
        todo!();
        /*
            return sub_(other);
        */
    }
}

impl SubAssign<Scalar> for Tensor {
    
    fn sub_assign(&mut self, other: &Scalar) -> Self::Output {
        todo!();
        /*
            return sub_(other);
        */
    }
}

impl MulAssign<&Tensor> for Tensor {
    
    fn mul_assign(&mut self, other: &Tensor) {
        todo!();
        /*
            return mul_(other);
        */
    }
}

impl MulAssign<Scalar> for Tensor {
    
    fn mul_assign(&mut self, other: Scalar) {
        todo!();
        /*
            return mul_(other);
        */
    }
}

impl DivAssign<&Tensor> for Tensor {
    
    fn div_assign(&mut self, other: &Tensor) {
        todo!();
        /*
            return div_(other);
        */
    }
}

impl DivAssign<Scalar> for Tensor {
    
    fn div_assign(&mut self, other: Scalar) {
        todo!();
        /*
            return div_(other);
        */
    }
}

impl BitAndAssign<&Tensor> for Tensor {
    
    fn bitand_assign(&mut self, other: &Tensor) {
        todo!();
        /*
            return bitwise_and_(other);
        */
    }
}

impl BitOrAssign<&Tensor> for Tensor {
    
    fn bitor_assign(&mut self, other: &Tensor) {
        todo!();
        /*
            return bitwise_or_(other);
        */
    }
}

impl BitXorAssign<&Tensor> for Tensor {
    
    fn bitxor_assign(&mut self, other: &Tensor) {
        todo!();
        /*
            return bitwise_xor_(other);
        */
    }
}

impl Index<Scalar> for Tensor {
    type Output = Tensor;
    
    fn index(&self, index: Scalar) -> &Self::Output {
        todo!();
        /*
            if (!index.isIntegral(false)) {
        TORCH_CHECK_INDEX(false, "Can only index tensors with integral scalars");
      }
      return select(0, index.toLong());
        */
    }
}

impl Index<Tensor> for Tensor {

    type Output = Tensor;
    
    fn index(&self, index: Tensor) -> &Self::Output {
        todo!();
        /*
            // These properties are checked in the Scalar constructor, but we already
      // check them here to provide more useful diagnostics for the user.
      if (!index.defined()) {
        TORCH_CHECK_INDEX(false, "Can only index with tensors that are defined");
      }
      if (index.dim() != 0) {
        TORCH_CHECK_INDEX(false,
          "Can only index with tensors that are scalars (zero-dim)");
      }
      // The Scalar(Tensor) constructor is explicit, so we need to call it.
      return this->operator[](index.item());
        */
    }
}

impl Index<i64> for Tensor {

    type Output = Tensor;
    
    fn index(&self, index: i64) -> &Self::Output {
        todo!();
        /*
            return select(0, index);
        */
    }
}

#[macro_export] macro_rules! at_forall_binary_ops {
    ($_:ident) => {
        /*
        
        _(+,x.add(y), y.add(x)) 
        _(*,x.mul(y), y.mul(x)) 
        _(-,x.sub(y), ::empty_like(y, MemoryFormat::Preserve).fill_(x).sub_(y)) 
        _(/,x.div(y), ::empty_like(y, MemoryFormat::Preserve).fill_(x).div_(y)) 
        _(%,x.remainder(y), ::empty_like(y, MemoryFormat::Preserve).fill_(x).remainder_(y)) 
        _(&,x.bitwise_and(y), y.bitwise_and(x)) 
        _(|,x.bitwise_or(y), y.bitwise_or(x)) 
        _(^,x.bitwise_xor(y), y.bitwise_xor(x)) 
        _(<,x.lt(y), y.gt(x)) 
        _(<=,x.le(y), y.ge(x)) 
        _(>,x.gt(y),y.lt(x)) 
        _(>=,x.ge(y), y.le(x)) 
        _(==,x.eq(y), y.eq(x)) 
        _(!=,x.ne(y), y.ne(x))
        */
    }
}

#[macro_export] macro_rules! define_operator {
    ($op:ident, $body:ident, $reverse_scalar_body:ident) => {
        /*
        
        static inline Tensor operator op(const Tensor & x, const Tensor & y) { 
          return body; 
        } 
        static inline Tensor operator op(const Tensor & x, const Scalar& y) { 
          return body; 
        } 
        static inline Tensor operator op(const Scalar& x, const Tensor & y) { 
          return reverse_scalar_body; 
        }
        */
    }
}

at_forall_binary_ops!{define_operator}
