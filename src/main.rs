// TODO: make this an example, not a main.
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
    let cache = Cache::from_bytes(&file)?;

    for font in cache.set()?.fonts()?.take(1) {
        let font = font?;

        for elt in font.elts()? {
            println!(
                "object type {:?}",
                Object::try_from(elt.data()?.object).unwrap()
            );

            for val in elt.values()? {
                let val = val?;
                if let fontconfig_cache_parser::Value::String(s) = val {
                    println!("string value: {:?}", String::from_utf8_lossy(s.str()?));
                } else if let fontconfig_cache_parser::Value::CharSet(cs) = val {
                    for ch in cs.chars()? {
                        println!("char {:x}", ch?);
                    }

                    println!("contains 0x63b? {}", cs.contains(0x63b)?);
                    println!("contains 0x6b5? {}", cs.contains(0x6b5)?);
                } else {
                    println!("val {:?}", val);
                }
            }
        }
    }

    Ok(())
}
