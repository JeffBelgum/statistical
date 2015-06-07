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

use num::{FromPrimitive,
          Float,
          Num,
          NumCast,
          One,
          Zero};

/// The mean is the sum of a collection of numbers divided by the number of numbers in the collection.
/// (reference)[http://en.wikipedia.org/wiki/Arithmetic_mean]
pub fn mean<T>(v: &[T]) -> T
    where T: Copy + FromPrimitive + Num + NumCast
{
    let len = FromPrimitive::from_usize(v.len()).unwrap();
    v.iter().fold(Zero::zero(), |acc: T, elem| acc + *elem) / len
}

/// The median is the number separating the higher half of a data sample, a population, or
/// a probability distribution, from the lower half (reference)[http://en.wikipedia.org/wiki/Median)
pub fn median<T>(v: &[T]) -> T
    where T: Copy + FromPrimitive + Num + NumCast + PartialOrd
{
    assert!(v.len() > 0);
    let mut scratch = Vec::with_capacity(v.len());
    scratch.extend(v.iter());
    quicksort(&mut scratch);

    let mid = scratch.len() / 2;
    if scratch.len() % 2 == 1 {
        *scratch[mid]
    } else {
        (*scratch[mid] + *scratch[mid-1]) / FromPrimitive::from_i32(2).unwrap()
    }
}

fn sum_square_deviations<T>(v: &[T], c: Option<T>) -> T
    where T: Copy + FromPrimitive + Num + NumCast + PartialOrd
{
    let c = match c {
        Some(c) => c,
        None => mean(v),
    };

    let sum = v.iter().map( |x| (*x - c) * (*x - c) ).fold(Zero::zero(), |acc, elem| acc + elem);
    assert!(sum >= Zero::zero(), "negative sum of square root deviations");
    sum
}

/// (Sample variance)[http://en.wikipedia.org/wiki/Variance#Sample_variance]
pub fn variance<T>(v: &[T], xbar: Option<T>) -> T
    where T: Copy + FromPrimitive + Num + NumCast + PartialOrd
{
    assert!(v.len() > 1, "variance requires at least two data points");
    let len: T = FromPrimitive::from_usize(v.len()).unwrap();
    let sum = sum_square_deviations(v, xbar);
    sum / (len - One::one())
}

/// (Population variance)[http://en.wikipedia.org/wiki/Variance#Population_variance]
pub fn population_variance<T>(v: &[T], mu: Option<T>) -> T
    where T: Copy + FromPrimitive + Num + NumCast + PartialOrd
{
    assert!(v.len() > 0, "population variance requires at least one data point");
    let len: T = FromPrimitive::from_usize(v.len()).unwrap();
    let sum = sum_square_deviations(v, mu);
    sum / len
}

///  Standard deviation is a measure that is used to quantify the amount of variation or
///  dispersion of a set of data values. (reference)[http://en.wikipedia.org/wiki/Standard_deviation]
pub fn standard_deviation<T>(v: &[T], xbar: Option<T>) -> T
    where T: Float + FromPrimitive
{
    let var = variance(v, xbar);
    var.sqrt()
}

pub fn population_standard_deviation<T>(v: &[T], mu: Option<T>) -> T
    where T: Float + FromPrimitive
{
    let pvar = population_variance(v, mu);
    pvar.sqrt()
}


/// Standard score is a given datum's (signed) number of standard deviations above the mean.
/// (reference)[http://en.wikipedia.org/wiki/Standard_score]
/// Method returns a vector of scores for a vector of inputs. scores[n] is the score of v[n]
pub fn standard_scores<T>(v: &[T]) -> Vec<T> where T: Float + FromPrimitive {
    let mean = mean(&v);
    let standard_deviation = standard_deviation(&v, None);
    let scores: Vec<T> = v.iter().map(|val| (*val - mean)/standard_deviation).collect();
    return scores;
}

#[inline(always)]
fn select_pivot<T>(v: &mut [T]) where T: Copy {
    let idx = rand::random::<usize>() % v.len();
    let tmp = v[0];
    v[0] = v[idx];
    v[idx] = tmp;
}

fn partition<T>(v: &mut [T]) -> usize where T: PartialOrd + Copy {
    select_pivot(v);
    let pivot = v[0];
    let mut i = 0;
    let mut j = 0;
    let end = v.len() - 1;
    while i < end {
        i += 1;
        if v[i] < pivot {
            v[j] = v[i];
            j += 1;
            v[i] = v[j];
        }

    }
    v[j] = pivot;
    j

}

fn quicksort<T>(v: &mut [T]) where T: PartialOrd + Copy {
    if v.len() <= 1 {
        return
    }
    let pivot = partition(v);
    quicksort(&mut v[..pivot]);
    quicksort(&mut v[(pivot+1)..]);
}

// Unit tests for private functions must be in the same module.
#[test]
fn test_qsort_empty() {
    let mut vec: Vec<f64> = vec![];
    quicksort(&mut vec);
    assert_eq!(vec, vec![]);
}

#[test]
fn test_qsort_small() {
    let len = 10;
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len { vec.push(rand::random::<f64>()); }
    quicksort(&mut vec);
    for i in 0..(len-1) {
        assert!(vec[i] < vec[i+1], "sorted vectors must be monotonically increasing");
    }
}

#[test]
fn test_qsort_large() {
    let len = 1_000_000;
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len { vec.push(rand::random::<f64>()); }
    quicksort(&mut vec);
    for i in 0..(len-1) {
        assert!(vec[i] < vec[i+1], "sorted vectors must be monotonically increasing");
    }
}

#[test]
fn test_qsort_sorted() {
    let len = 1_000;
    let mut vec = Vec::with_capacity(len);
    for n in 0..len { vec.push(n); }
    quicksort(&mut vec);
    for i in 0..(len-1) {
        assert!(vec[i] < vec[i+1], "sorted vectors must be monotonically increasing");
    }
}

#[test]
fn test_qsort_reverse_sorted() {
    let len = 1_000;
    let mut vec = Vec::with_capacity(len);
    for n in 0..len { vec.push(len-n); }
    quicksort(&mut vec);
    for i in 0..(len-1) {
        assert!(vec[i] < vec[i+1], "sorted vectors must be monotonically increasing");
    }
}


#[cfg(test)]
mod tests {
    extern crate rand;
    extern crate num;

    use super::*;
    use num::Float;

    #[test]
    fn test_mean() {
        let vec = vec![0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25];
        assert_eq!(mean(&vec), 1.375);
    }

    #[test]
    fn test_median() {
        let vec = vec![1.0, 3.0];
        assert_eq!(median(&vec), 2.0);
        let vec = vec![1.0, 3.0, 5.0];
        assert_eq!(median(&vec), 3.0);
        let vec = vec![1.0, 3.0, 5.0, 7.0];
        assert_eq!(median(&vec), 4.0);
    }

    #[test]
    fn test_variance() {
        let v = vec![0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25];
        // result is within `epsilon` of expected value
        let expected = 1.428571;
        let epsilon = 1e-6;
        assert!((expected - variance(&v, None)).abs() < epsilon);
    }

    #[test]
    fn test_population_variance() {
        let v = vec![0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25];
        // result is within `epsilon` of expected value
        let expected = 1.25;
        let epsilon = 1e-6;
        assert!((expected - population_variance(&v, None)).abs() < epsilon);
    }

    #[test]
    fn test_standard_deviation() {
        let v = vec![0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25];
        // result is within `epsilon` of expected value
        let expected = 1.195229;
        let epsilon = 1e-6;
        assert!((expected - standard_deviation(&v, None)).abs() < epsilon);
    }

    #[test]
    fn test_population_standard_deviation() {
        let v = vec![0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25];
        // result is within `epsilon` of expected value
        let expected = 1.118034;
        let epsilon = 1e-6;
        assert!((expected - population_standard_deviation(&v, None)).abs() < epsilon);
    }

    #[test]
    fn test_standard_scores() {
        let v = vec![0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25];
        let expected = vec![-1.150407536484354, -0.941242529850835, -0.941242529850835, -0.10458250331675945, 0.10458250331675945, 0.31374750995027834, 1.150407536484354, 1.5687375497513918];
        assert!(expected == standard_scores(&v));
    }
}
