#![allow(dead_code)]

use crate::types::{CVector, CVectorView};

pub struct LinearOperator {
    pub shape: (usize, usize),
    pub matvec: Box<dyn Fn(CVectorView) -> CVector>,
}
