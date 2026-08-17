use feathertalk_parity::archive::GoldenArchive;
use std::{fs::File, io::Write};
use zip::{ZipWriter, write::SimpleFileOptions};

fn golden_archive() -> GoldenArchive {
    let root = env!("CARGO_MANIFEST_DIR");
    GoldenArchive::open(format!("{root}/../../tests/golden/burn-feasibility-v1.zip"))
        .expect("golden archive should open")
}

#[test]
fn golden_archive_has_required_entries_and_valid_hash() {
    let archive = golden_archive();

    archive.verify_sidecar_sha256().expect("archive hash");
    for entry in [
        "manifest.json",
        "weights/tiny_direct.pth",
        "weights/tiny_nested.pth",
        "weights/tiny_missing.pth",
        "weights/tiny_unexpected.pth",
        "weights/feather_micro.pth",
        "weights/unet_production.pth",
        "weights/unet_micro_train.pth",
        "arrays/feather_input.npy",
        "arrays/feather_output.npy",
        "arrays/unet_image.npy",
        "arrays/unet_audio.npy",
        "arrays/unet_output.npy",
        "arrays/train_target.npy",
        "arrays/train_expected.json",
    ] {
        assert!(archive.contains(entry), "missing {entry}");
    }
}

#[test]
fn feather_fixture_loads_named_arrays_with_expected_shapes() {
    let fixture = golden_archive()
        .load_fixture("feather_micro_eval")
        .expect("fixture should load");

    assert_eq!(fixture.id, "feather_micro_eval");
    assert_eq!(fixture.inputs["waveform"].shape(), &[1, 1360]);
    assert_eq!(fixture.expected["output"].shape(), &[1, 4, 64]);
}

#[test]
fn extraction_rejects_parent_traversal() {
    let temp = tempfile::tempdir().unwrap();
    let archive_path = temp.path().join("malicious.zip");
    let file = File::create(&archive_path).unwrap();
    let mut writer = ZipWriter::new(file);
    writer
        .start_file("../escape.txt", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(b"escaped").unwrap();
    writer.finish().unwrap();

    let destination = temp.path().join("unpacked");
    let archive = GoldenArchive::open(&archive_path).unwrap();
    assert!(archive.extract_to(&destination).is_err());
    assert!(!temp.path().join("escape.txt").exists());
}
