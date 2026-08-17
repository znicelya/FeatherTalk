use ndarray::ArrayD;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct GoldenFixture {
    pub id: String,
    pub schema_version: u32,
    pub fixture_set: String,
    pub kind: String,
    pub weights_entry: String,
    pub config: BTreeMap<String, Value>,
    pub optimizer: Option<BTreeMap<String, Value>>,
    pub loss: Option<String>,
    pub expected_mode: Option<String>,
    pub inputs: BTreeMap<String, ArrayD<f32>>,
    pub expected: BTreeMap<String, ArrayD<f32>>,
    pub metrics: BTreeMap<String, f64>,
    pub scalars: BTreeMap<String, f64>,
}
