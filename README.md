# fontconfig-cache-parser

`fontconfig-cache-parser` does what it says: it's a rust crate for parsing fontconfig's cache files.
This allows you to list your installed font files (and see some properties of them) without having
to scan all the font directories on your system.

This crate does not (yet) allow for parsing all the information in the cache, but it is capable of
extracting basic metadata like filenames, style information, and charsets.

You may also be interested in the [`fontconfig_parser`](https://github.com/Riey/fontconfig-parser) crate,
which parses fontconfig's config files. In particular, it can tell you where the cache files live.