use clap::Parser;
use std::path::PathBuf;

use fontconfig_cache_parser::Cache;

#[derive(Parser, Debug)]
struct Args {
    /// Path to a fontconfig cache file.
    path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let file = std::fs::read(args.path)?;
    let cache = Cache::read(&file)?;
    println!("read header: {:?}", cache);

    println!(
        "dir {:?}",
        String::from_utf8_lossy(cache.decode_str(&file, cache.payload.dir).unwrap())
    );

    let set = cache.decode(&file, cache.payload.set)?;
    println!("read set {:?}", set);

    for font in set.fonts(&file)?.take(10) {
        println!("read font {:?}", font?);
    }

    if let Some(first_font) = set.fonts(&file)?.next() {
        let first_font = first_font?;
        println!("first font {:?}", first_font);

        for elt in first_font.elts(&file)?.take(4) {
            println!("elt {:?}", elt);

            for val in elt.values(&file)? {
                let val = val?.to_enum()?;
                println!("val {:?}", val);
                if let fontconfig_cache_parser::ValueEnum::String(offset) = val.payload {
                    println!(
                        "string value: {:?}",
                        String::from_utf8_lossy(val.decode_str(&file, offset)?)
                    );
                }
            }
        }
    }

    Ok(())
}
