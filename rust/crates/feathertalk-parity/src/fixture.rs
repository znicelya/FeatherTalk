use ndarray::ArrayD;
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct GoldenFixture {
    pub id: String,
    pub inputs: BTreeMap<String, ArrayD<f32>>,
    pub expected: BTreeMap<String, ArrayD<f32>>,
}
