crate::ix!();

pub struct ProfDAGStats {
    sum:     f32,
    sqrsum:  f32,
    cnt:     usize,
}

impl Default for ProfDAGStats {
    
    fn default() -> Self {
        todo!();
        /*
            : sum_(0.0), sqrsum_(0.0), cnt_(0
        */
    }
}

impl AddAssign<&ProfDAGStats> for ProfDAGStats {
    
    fn add_assign(&mut self, other: &ProfDAGStats) {
        todo!();
        /*
            sum_ += rhs.sum_;
        sqrsum_ += rhs.sqrsum_;
        cnt_ += rhs.cnt_;
        return *this;
        */
    }
}

impl ProfDAGStats {
    
    pub fn new(time_ms: f32) -> Self {
    
        todo!();
        /*
            : sum_(time_ms), sqrsum_(time_ms * time_ms), cnt_(1)
        */
    }
    
    #[inline] pub fn compute_moments(&self) -> (f32,f32) {
        
        todo!();
        /*
            CAFFE_ENFORCE_GT(cnt_, 0U);
        float mean = sum_ / cnt_;
        float stddev = std::sqrt(std::abs(sqrsum_ / cnt_ - mean * mean));
        return {mean, stddev};
        */
    }
    
    #[inline] pub fn sum(&self) -> f32 {
        
        todo!();
        /*
            return sum_;
        */
    }
    
    #[inline] pub fn sqrsum(&self) -> f32 {
        
        todo!();
        /*
            return sqrsum_;
        */
    }
    
    #[inline] pub fn cnt(&self) -> usize {
        
        todo!();
        /*
            return cnt_;
        */
    }
}
