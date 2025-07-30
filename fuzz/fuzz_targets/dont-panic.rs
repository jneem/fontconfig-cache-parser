#![no_main]
use fontconfig_cache_parser::Cache;
use libfuzzer_sys::fuzz_target;

fn run(data: &[u8]) -> anyhow::Result<()> {
    let cache = Cache::from_bytes(data)?;

    for font in cache.set()?.fonts()? {
        let font = font?;

        for elt in font.elts()? {
            let _ = elt.data()?;

            // `values` is a linked list, so it could be a cycle. Truncate it.
            for val in elt.values()?.take(5) {
                let val = val?;
                if let fontconfig_cache_parser::Value::String(s) = val {
                    let _ = s.str()?;
                } else if let fontconfig_cache_parser::Value::CharSet(cs) = val {
                    for ch in cs.chars()? {
                        let _ = ch?;
                    }

                    let _ = cs.contains(0x63b)?;
                }
            }
        }
    }
    Ok(())
}

fuzz_target!(|data: &[u8]| {
    let _ = run(data);
});
