# macOS AppleDouble sidecars

`appledouble_sidecars_765ded6.tar.gz` holds the 77 `._*` files that commit
`765ded6` removed, at their original repository paths.

They are AppleDouble encoded Macintosh files: Finder resource-fork metadata that
macOS writes beside real files when copying onto a non-HFS filesystem. They carry
no project content -- each is a 4096-byte metadata block, which is why all 77
compress to 4 KB.

They are kept as a tarball rather than as loose files because `.gitignore:43`
matches `._*`, so loose copies here would be untracked and could be lost again.

A further 268 `._*` files were deleted from the working tree on 2026-09-09.
Those were never tracked by git, so they could not be recovered and are not in
this archive.
