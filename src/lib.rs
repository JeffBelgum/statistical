// Copyright (c) 2015 Jeff Belgum
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without
// limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

//! A simple statistics library
//!
//! Heavily inspired by the python standard library statistics module.

extern crate rand;
extern crate num;

mod univariate_;
mod stats_;

pub mod univariate {
    pub use univariate_::{
        harmonic_mean,
        geometric_mean,
        quadratic_mean,
        mode,
        average_deviation,
        pearson_skewness,
        skewness,
        pskewness,
        kurtosis,
        pkurtosis,
        standard_error_mean,
        standard_error_skewness,
        standard_error_kurtosis
    };
}

pub use univariate::mode;
pub use stats_::{
    Degree,
    mean,
    median,
    variance,
    population_variance,
    standard_deviation,
    population_standard_deviation,
    standard_scores,
    percentile,
};
