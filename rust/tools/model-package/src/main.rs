use std::path::PathBuf;

use clap::{Parser, Subcommand};
use feathertalk_export::{
    FeatherHubertPackageRequest, ModelConfiguration, build_feather_hubert_package,
};

#[derive(Debug, Parser)]
#[command(
    name = "feathertalk-model-package",
    about = "Build auditable FeatherTalk model packages"
)]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(name = "feather-hubert")]
    FeatherHubert {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        licenses: PathBuf,
        #[arg(long)]
        destination: PathBuf,
        #[arg(long)]
        created_at: String,
        #[arg(long)]
        minimum_app_version: String,
    },
}

fn main() -> Result<(), feathertalk_export::PackageError> {
    let arguments = Arguments::parse();
    match arguments.command {
        Command::FeatherHubert {
            source,
            licenses,
            destination,
            created_at,
            minimum_app_version,
        } => {
            let report = build_feather_hubert_package(&FeatherHubertPackageRequest {
                source,
                licenses,
                destination: destination.clone(),
                created_at,
                minimum_app_version,
            })?;
            println!("destination={}", destination.display());
            println!("source_sha256={}", report.manifest.source.sha256);
            println!("model_sha256={}", report.manifest.model.sha256);
            println!("tensor_count={}", report.manifest.tensors.tensor_count);
            println!("total_elements={}", report.manifest.tensors.total_elements);
            if let ModelConfiguration::FeatherHubert {
                channels,
                expansion,
                num_blocks,
                output_dim,
                dropout,
            } = report.manifest.configuration
            {
                println!(
                    "configuration=channels:{channels},expansion:{expansion},num_blocks:{num_blocks},output_dim:{output_dim},dropout:{dropout}"
                );
            }
        }
    }
    Ok(())
}
