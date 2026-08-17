use burn_store::{KeyRemapper, PytorchStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyModelKind {
    FeatherHubert,
    OriginalUnet,
}

pub fn is_known_ignored_key(key: &str) -> bool {
    key.ends_with(".num_batches_tracked")
}

pub(crate) fn configure_store(
    path: impl Into<std::path::PathBuf>,
    kind: LegacyModelKind,
    top_level_key: Option<&str>,
) -> PytorchStore {
    let store = PytorchStore::from_file(path)
        .allow_partial(true)
        .validate(false)
        .map_indices_contiguous(false)
        .remap(remapper_for(kind));

    match top_level_key {
        Some(key) => store.with_top_level_key(key),
        None => store,
    }
}

pub(crate) fn map_key(kind: LegacyModelKind, key: &str) -> String {
    let mut mapped = key.to_owned();
    for (pattern, replacement) in remapper_for(kind).patterns {
        mapped = pattern
            .replace_all(&mapped, replacement.as_str())
            .into_owned();
    }
    mapped
}

fn remapper_for(kind: LegacyModelKind) -> KeyRemapper {
    match kind {
        LegacyModelKind::FeatherHubert => KeyRemapper::new()
            .add_pattern(r"((?:^|\.)(?:norm|final_norm))\.weight$", "$1.gamma")
            .expect("reviewed literal regex")
            .add_pattern(r"((?:^|\.)(?:norm|final_norm))\.bias$", "$1.beta")
            .expect("reviewed literal regex"),
        LegacyModelKind::OriginalUnet => KeyRemapper::new()
            .add_pattern(r"\.double_conv\.0\.", ".first.")
            .expect("reviewed literal regex")
            .add_pattern(r"\.double_conv\.1\.", ".second.")
            .expect("reviewed literal regex")
            .add_pattern(r"\.conv\.0\.", ".expand_conv.")
            .expect("reviewed literal regex")
            .add_pattern(r"\.conv\.1\.", ".expand_bn.")
            .expect("reviewed literal regex")
            .add_pattern(r"\.conv\.3\.", ".depthwise_conv.")
            .expect("reviewed literal regex")
            .add_pattern(r"\.conv\.4\.", ".depthwise_bn.")
            .expect("reviewed literal regex")
            .add_pattern(r"\.conv\.6\.", ".project_conv.")
            .expect("reviewed literal regex")
            .add_pattern(r"\.conv\.7\.", ".project_bn.")
            .expect("reviewed literal regex")
            .add_pattern(
                r"(\.(?:expand_bn|depthwise_bn|project_bn|bn3|bn5))\.weight$",
                "$1.gamma",
            )
            .expect("reviewed literal regex")
            .add_pattern(
                r"(\.(?:expand_bn|depthwise_bn|project_bn|bn3|bn5))\.bias$",
                "$1.beta",
            )
            .expect("reviewed literal regex")
            .add_pattern(r"^fuse_conv\.0\.", "fuse_first.")
            .expect("reviewed literal regex")
            .add_pattern(r"^fuse_conv\.1\.", "fuse_second.")
            .expect("reviewed literal regex"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_unet_keys_follow_reviewed_remaps() {
        let cases = [
            (
                "down1.maxpool_conv.double_conv.0.conv.0.weight",
                "down1.maxpool_conv.first.expand_conv.weight",
            ),
            (
                "down1.maxpool_conv.double_conv.1.conv.7.running_var",
                "down1.maxpool_conv.second.project_bn.running_var",
            ),
            (
                "fuse_conv.0.double_conv.0.conv.3.weight",
                "fuse_first.first.depthwise_conv.weight",
            ),
            (
                "fuse_conv.1.double_conv.1.conv.6.weight",
                "fuse_second.second.project_conv.weight",
            ),
            (
                "down1.maxpool_conv.double_conv.0.conv.1.weight",
                "down1.maxpool_conv.first.expand_bn.gamma",
            ),
            ("audio_model.bn3.bias", "audio_model.bn3.beta"),
        ];

        for (source, expected) in cases {
            assert_eq!(map_key(LegacyModelKind::OriginalUnet, source), expected);
        }
    }

    #[test]
    fn feather_hubert_keys_are_unchanged() {
        assert_eq!(
            map_key(LegacyModelKind::FeatherHubert, "encoder.0.dw_conv.weight"),
            "encoder.0.dw_conv.weight"
        );
    }

    #[test]
    fn feather_hubert_group_norm_parameters_use_burn_names() {
        let cases = [
            (
                "frontend.layers.0.norm.weight",
                "frontend.layers.0.norm.gamma",
            ),
            ("frontend.layers.0.norm.bias", "frontend.layers.0.norm.beta"),
            ("final_norm.weight", "final_norm.gamma"),
            ("final_norm.bias", "final_norm.beta"),
        ];

        for (source, expected) in cases {
            assert_eq!(map_key(LegacyModelKind::FeatherHubert, source), expected);
        }
    }
}
