use std::sync::{Arc, Mutex};

use feathertalk_audio::{
    AudioError, ChunkEncoder, DEFAULT_CHUNK_SAMPLES, FeatureMatrix, drop_odd_token,
    extract_long_audio,
};

#[derive(Clone)]
struct FakeEncoder {
    dim: usize,
    calls: Arc<Mutex<Vec<(usize, usize, f32)>>>,
    rows_per_call: Vec<usize>,
}

impl ChunkEncoder for FakeEncoder {
    fn output_dim(&self) -> usize {
        self.dim
    }

    fn encode(&mut self, chunk_index: usize, samples: &[f32]) -> Result<Vec<f32>, AudioError> {
        self.calls
            .lock()
            .unwrap()
            .push((chunk_index, samples.len(), samples[0]));
        let rows = self.rows_per_call[chunk_index];
        Ok((0..rows)
            .flat_map(|row| std::iter::repeat_n((chunk_index * 100 + row) as f32, self.dim))
            .collect())
    }
}

fn waveform(length: usize) -> Vec<f32> {
    (0..length).map(|value| value as f32).collect()
}

#[test]
fn extracts_in_chunk_order_and_uses_extended_full_chunk_ranges() {
    let samples = waveform(DEFAULT_CHUNK_SAMPLES + 720);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut encoder = FakeEncoder {
        dim: 2,
        calls: calls.clone(),
        rows_per_call: vec![4, 2],
    };
    let matrix = extract_long_audio(&samples, &mut encoder, DEFAULT_CHUNK_SAMPLES).unwrap();
    assert_eq!(matrix.tokens(), 1002);
    assert_eq!(matrix.dims(), 2);
    assert_eq!(
        calls.lock().unwrap().as_slice(),
        &[
            (0, DEFAULT_CHUNK_SAMPLES + 80, 0.0),
            (1, 720, DEFAULT_CHUNK_SAMPLES as f32),
        ]
    );
    assert_eq!(
        &matrix.values()[..12],
        &[
            0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 100.0, 100.0, 101.0, 101.0
        ]
    );
    assert!(matrix.values()[12..].iter().all(|value| *value == 0.0));
}

#[test]
fn crops_long_encoder_output_pads_short_output_and_drops_odd_token() {
    let samples = waveform(1360);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut short = FakeEncoder {
        dim: 2,
        calls,
        rows_per_call: vec![2],
    };
    let padded = extract_long_audio(&samples, &mut short, DEFAULT_CHUNK_SAMPLES).unwrap();
    assert_eq!(padded.tokens(), 4);
    assert_eq!(padded.values(), &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    let even = drop_odd_token(FeatureMatrix::new(5, 2, vec![1.0; 10]).unwrap());
    assert_eq!(even.tokens(), 4);
    assert_eq!(even.values().len(), 8);
}

#[test]
fn rejects_encoder_dimension_length_and_non_finite_output() {
    struct Bad;
    impl ChunkEncoder for Bad {
        fn output_dim(&self) -> usize {
            2
        }
        fn encode(&mut self, _: usize, _: &[f32]) -> Result<Vec<f32>, AudioError> {
            Ok(vec![f32::NAN; 2])
        }
    }
    let mut bad = Bad;
    assert!(matches!(
        extract_long_audio(&waveform(720), &mut bad, DEFAULT_CHUNK_SAMPLES),
        Err(AudioError::NonFiniteFeature { .. })
    ));

    struct WrongLength;
    impl ChunkEncoder for WrongLength {
        fn output_dim(&self) -> usize {
            4
        }
        fn encode(&mut self, _: usize, _: &[f32]) -> Result<Vec<f32>, AudioError> {
            Ok(vec![0.0; 3])
        }
    }
    assert!(matches!(
        extract_long_audio(&waveform(720), &mut WrongLength, DEFAULT_CHUNK_SAMPLES),
        Err(AudioError::FeatureLengthMismatch {
            actual: 3,
            dimension: 4
        })
    ));
}
