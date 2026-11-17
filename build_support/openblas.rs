//! GEADD changed ABI in OpenBLAS 0.3.34, without changing symbol names.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeaddAbi {
    Legacy,
    Transpose,
}

/// Inspect all four declarations. Ignore comments, which can contain examples.
pub fn header_abi(header: &str) -> Result<GeaddAbi, String> {
    let mut clean = String::new();
    let mut chars = header.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '/' && chars.peek() == Some(&'*') {
            chars.next();
            while let Some(c) = chars.next() {
                if c == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    break;
                }
            }
            clean.push(' ');
        } else if c == '/' && chars.peek() == Some(&'/') {
            for c in chars.by_ref() {
                if c == '\n' {
                    break;
                }
            }
            clean.push('\n');
        } else {
            clean.push(c);
        }
    }
    let mut detected = None;
    for name in [
        "cblas_sgeadd",
        "cblas_dgeadd",
        "cblas_cgeadd",
        "cblas_zgeadd",
    ] {
        let args = clean
            .match_indices(name)
            .find_map(|(start, _)| clean[start + name.len()..].trim_start().strip_prefix('('))
            .ok_or_else(|| format!("missing {name} declaration"))?
            .split_once(')')
            .ok_or_else(|| format!("unterminated {name} declaration"))?
            .0;
        let abi = match args.split(',').count() {
            9 => GeaddAbi::Legacy,
            11 => GeaddAbi::Transpose,
            n => return Err(format!("unrecognized {name} signature ({n} arguments)")),
        };
        if detected.is_some_and(|previous| previous != abi) {
            return Err("inconsistent GEADD declarations".into());
        }
        detected = Some(abi);
    }
    Ok(detected.unwrap())
}

/// Used only when the discovered installation has no readable CBLAS header.
pub fn version_abi(version: &str) -> Result<GeaddAbi, String> {
    let invalid = || format!("invalid OpenBLAS version: {version}");
    let mut parts = version.splitn(3, '.');
    let major: u32 = parts
        .next()
        .ok_or_else(invalid)?
        .parse()
        .map_err(|_| invalid())?;
    let minor: u32 = parts
        .next()
        .ok_or_else(invalid)?
        .parse()
        .map_err(|_| invalid())?;
    let patch = parts.next().ok_or_else(invalid)?;
    let digits = patch.bytes().take_while(u8::is_ascii_digit).count();
    let suffix = &patch[digits..];
    if !suffix.is_empty() && !suffix.starts_with(['.', '-', '+']) {
        return Err(invalid());
    }
    let patch: u32 = patch[..digits].parse().map_err(|_| invalid())?;
    let version = (major, minor, patch);
    Ok(if version >= (0, 3, 34) {
        GeaddAbi::Transpose
    } else {
        GeaddAbi::Legacy
    })
}
