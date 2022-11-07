use clap::Parser;
use std::path::PathBuf;

use fontconfig_cache_parser::{Cache, Object};

#[derive(Parser, Debug)]
struct Args {
    /// Path to a fontconfig cache file.
    path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let file = std::fs::read(args.path)?;
    let cache = Cache::read(&file)?;
    let set = cache.decode(cache.payload.set)?;

    for font in set.fonts()?.take(3) {
        let font = font?;
        println!("font {:?}", font.payload);

        for elt in font.elts()? {
            println!(
                "object type {:?}",
                Object::try_from(elt.payload.object).unwrap()
            );

            for val in elt.values()? {
                let val = val?.to_enum()?;
                if let fontconfig_cache_parser::ValueEnum::String(offset) = val.payload {
                    println!(
                        "string value: {:?}",
                        String::from_utf8_lossy(val.decode_str(offset)?)
                    );
                } else {
                    println!("val {:?}", val.payload);
                }
            }
        }
    }

    Ok(())
}
