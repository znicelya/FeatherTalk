# SCRFD Burn importer

This standalone tool validates `data_utils/scrfd_2.5g_kps.onnx`, generates the
pinned Burn 0.21 graph, converts its temporary burnpack state to safetensors,
and publishes a deterministic four-file artifact tree. The model source hash
is `32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a` and the
tracked ONNX file is 3,291,017 bytes.

Run from `rust/`:

```powershell
cargo run --manifest-path tools/scrfd-import/Cargo.toml --bin generate -- --repo-root .. --destination crates/feathertalk-scrfd/.generated-candidate
cargo run --manifest-path tools/scrfd-import/Cargo.toml --bin generate -- --repo-root .. --verify-against crates/feathertalk-scrfd
```

The `.bpk` file exists only in a temporary staging directory and is never
published. The final tree contains generated Rust source, a source/weight hash
contract, safetensors weights, and the validated manifest. The manifest keeps
license status `NOASSERTION` with `redistribution_approved: false`.

The Python/OpenCV fixture generator is separate and is only used for Task 5:

```powershell
tools\scrfd-import\.venv\Scripts\python.exe tools\scrfd-import\python\generate_fixture.py --repo-root .. --verify-against crates\feathertalk-scrfd\tests\fixtures\opencv_cpu_v1
```
