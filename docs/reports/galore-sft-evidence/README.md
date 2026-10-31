# Complete reviewer evidence

Nine run archives and supporting.tar.xz retain all original raw files. Each archive is below the repository's 1000 KiB per-file limit. The manifest records every compressed and uncompressed SHA256 and length.

Run `python verify.py` here before extraction. To extract all files: `for archive in *.tar.xz; do tar -xJf "$archive"; done`. Archive member paths are relative; original local paths in metadata are provenance. Weights and full Dolly source rows are intentionally excluded; use the pinned portable reproduction instructions.
