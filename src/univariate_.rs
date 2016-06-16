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

extern crate rand;
extern crate num;

use std::collections::HashMap;
use std::hash::Hash;

use num::{Float,
          One,
          PrimInt,
          Zero};

use super::stats_ as stats;

pub fn harmonic_mean<T>(v: &[T]) -> T
    where T: Float
{
    let invert = |x: &T| T::one() / *x;
    let sum_of_inverted = v.iter().map(invert).fold(T::zero(), |acc, elem| acc + elem);
    num::cast::<usize, T>(v.len()).unwrap() / sum_of_inverted
}

pub fn geometric_mean<T>(v: &[T]) -> T
    where T: Float
{
    let product = v.iter().fold(T::one(), |acc, elem| acc * *elem);
    let one_over_len = T::one() / num::cast(v.len()).unwrap();
    product.powf(one_over_len)
}

pub fn quadratic_mean<T>(v: &[T]) -> T
    where T: Float
{
    let square = |x: &T| (*x).powi(2);
    let sum_of_squared = v.iter().map(square).fold(T::zero(), |acc, elem| acc + elem);
    (sum_of_squared / num::cast(v.len()).unwrap()).sqrt()
}

pub fn mode<C>(v: C) -> Option<C::Item>
    where C : IntoIterator,
    C::Item: Hash + Eq
{
    let mut counter = HashMap::new();
    let it = v.into_iter();
    for x in it {
        let count = counter.entry(x).or_insert(0);
        *count += 1;
    }
    let result = counter.drain().max_by_key(|&(_,count)| count);
    match result{
        None => None,
        Some((element, _)) => Some(element)
    }
}

pub fn average_deviation<T>(v: &[T], mean: Option<T>) -> T
    where T: Float
{
    let mean = mean.unwrap_or_else(|| stats::mean(v));
    let dev = v.iter().map(|&x| (x-mean).abs()).fold(T::zero(), |acc, elem| acc + elem);
    dev / num::cast(v.len()).unwrap()
}

pub fn pearson_skewness<T>(mean: T, mode: T, stdev: T) -> T
    where T: Float
{
    (mean - mode) / stdev
}

pub fn skewness<T>(v: &[T], mean: Option<T>, pstdev: Option<T>) -> T
    where T: Float
{
    let m = stats::std_moment(v, stats::Degree::Three, mean, pstdev);
    let n = num::cast(v.len()).unwrap();
    let skew = m / n;
    let k = ( n * ( n - T::one())).sqrt()/( n - num::cast(2).unwrap());
    skew * k
}

pub fn pskewness<T>(v: &[T], mean: Option<T>, pstdev: Option<T>) -> T
    where T: Float
{
    let m = stats::std_moment(v, stats::Degree::Three, mean, pstdev);
    m / num::cast(v.len()).unwrap()
}

pub fn kurtosis<T>(v: &[T], mean: Option<T>, pstdev: Option<T>) -> T
    where T: Float
{
    let two = num::cast::<f32, T>(2.0).unwrap();
    let three = num::cast::<f32, T>(3.0).unwrap();

    let m = stats::std_moment(v, stats::Degree::Four, mean, pstdev);
    let n = num::cast(v.len()).unwrap();
    let q = (n - T::one())/((n-two)*(n-three));
    let gamma2 = m / n;
    let kurt = q * (( ( n + T::one() ) * gamma2) - ( (n-T::one()) * three ));
    kurt
}

pub fn pkurtosis<T>(v: &[T], mean: Option<T>, pstdev: Option<T>) -> T
    where T: Float
{
    let m = stats::std_moment(v, stats::Degree::Four, mean, pstdev);
    m / num::cast(v.len()).unwrap() - num::cast(3).unwrap()
}

pub fn standard_error_mean<T>(stdev: T, sample_size: T, population_size: Option<T>) -> T
    where T: Float
{
    let mut err = stdev / sample_size.sqrt();
    if let Some(p) = population_size {
        err = err * ((p - sample_size) / (p - T::one())).sqrt()
    }
    err

}

pub fn standard_error_skewness<T, U>(sample_size: T) -> U
    where T: PrimInt, U: Float
{
    (num::cast::<f32,U>(6.0).unwrap() / num::cast(sample_size).unwrap()).sqrt()
}

pub fn standard_error_kurtosis<T, U>(sample_size: T) -> U
    where T: PrimInt, U: Float
{
    (num::cast::<f32,U>(24.0).unwrap() / num::cast(sample_size).unwrap()).sqrt()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_harmonic_mean() {
        let vec = vec![0.25, 0.5, 1.0, 1.0];
        assert_eq!(harmonic_mean(&vec), 0.5);
        let vec = vec![0.5, 0.5, 0.5];
        assert_eq!(harmonic_mean(&vec), 0.5);
        let vec = vec![1.0,2.0,4.0];
        assert_eq!(harmonic_mean(&vec), 12.0/7.0);
    }
    #[test]
    fn test_geometric_mean() {
        let vec = vec![1.0, 2.0, 6.125, 12.25];
        assert_eq!(geometric_mean(&vec), 3.5);
    }
    #[test]
    fn test_quadratic_mean() {
        let vec = vec![-3.0, -2.0, 0.0, 2.0, 3.0];
        assert_eq!(quadratic_mean(&vec), 2.280350850198276);
    }
    #[test]
    fn test_mode() {
        let vec = vec![2,4,3,5,4,6,1,1,6,4,0,0];
        assert_eq!(mode(&vec), Some(&4));
        let vec = vec![1];
        assert_eq!(mode(&vec), Some(&1));
        let vec = vec![2,4,3,5,4,6,1,1,6,4,0,0];
        assert_eq!(mode(vec), Some(4));
        let vec = vec![1,1,1,4];
        assert_eq!(mode(vec.iter().skip(3)), Some(&4));
    }

    #[derive(PartialEq, Eq, Hash, Debug)]
    enum Color {Blue, Green, Yellow}

    #[test]
    fn test_mode_of_colors(){
        assert_eq!(Some(&Color::Green), mode(&[Color::Blue, Color::Green, Color::Yellow, Color::Green]));
    }

    #[test]
    fn test_average_deviation() {
        let vec = vec![2.0, 2.25, 2.5, 2.5, 3.25];
        assert_eq!(average_deviation(&vec, None), 0.3);
        assert_eq!(average_deviation(&vec, Some(2.75)), 0.45);
    }
    #[test]
    fn test_pearson_skewness() {
        assert_eq!(pearson_skewness(2.5, 2.25, 2.5), 0.1);
        assert_eq!(pearson_skewness(2.5, 5.75, 0.5), -6.5);
    }
    #[test]
    fn test_skewness() {
        let vec = vec![1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5];
        assert_eq!(skewness(&vec, None, None), 1.7146101353987853);
        let vec = vec![1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5];
        assert_eq!(skewness(&vec, Some(2.25), Some(1.0)), 1.4713288161532945);
    }
    #[test]
    fn test_pskewness() {
        let vec = vec![1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5];
        assert_eq!(pskewness(&vec, None, None), 1.3747465025469285);
    }
    #[test]
    fn test_kurtosis() {
        let vec = vec![1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5];
        assert_eq!(kurtosis(&vec, None, None), 3.036788927335642);
        let vec = vec![1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5];
        assert_eq!(kurtosis(&vec, Some(2.25), Some(1.0)), 2.3064453125);
    }
    #[test]
    fn test_pkurtosis() {
        let vec = vec![1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5];
        assert_eq!(pkurtosis(&vec, None, None), 0.7794232987312579);
    }
    #[test]
    fn test_standard_error_mean() {
        assert_eq!(standard_error_mean(2.0, 16.0, None), 0.5);
    }
    #[test]
    fn test_standard_error_skewness() {
        assert_eq!(standard_error_skewness::<i32, f32>(15), 0.63245553203);
    }
    #[test]
    fn test_standard_error_kurtosis() {
        assert_eq!(standard_error_kurtosis::<i32, f32>(15), 1.2649110640);
    }
}
