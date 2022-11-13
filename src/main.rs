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

    for font in cache.set()?.fonts()?.take(1) {
        let font = font?;
        println!("font {:?}", font.payload());

        for elt in font.elts()? {
            println!(
                "object type {:?}",
                Object::try_from(elt.payload().object).unwrap()
            );

            for val in elt.values()? {
                let val = val?.to_value()?;
                if let fontconfig_cache_parser::Value::String(s) = val {
                    println!("string value: {:?}", String::from_utf8_lossy(s.str()?));
                } else if let fontconfig_cache_parser::Value::CharSet(cs) = val {
                    for chunk in cs.chunks()? {
                        println!("char set chunk: {:?}", chunk);
                    }
                } else {
                    println!("val {:?}", val);
                }
            }
        }
    }

    Ok(())
}
