/*
 * faber
 *
 * Copyright (C) 2025 Giuseppe Scrivano <giuseppe@scrivano.org>
 * faber is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * faber is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with faber.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

//! Files tasks produce (artifacts), kept by whoever has the database: its
//! rows (`db::ArtifactRow`) say which task made which file, under what
//! name; the bytes are in a directory, one file per content, named by its
//! digest:
//!
//! ```text
//! <dir>/sha256/<hex>    the contents, read-only
//! <dir>/tmp/<upload>    uploads not finished yet
//! <dir>/lock            taken to commit an upload, or to collect garbage
//! ```
//!
//! Agents don't write there: they may run elsewhere, connected to a
//! server. They upload through `DbBackend`, a chunk at a time
//! (`upload`), and the database's side commits it (`ArtifactStore`).

use crate::db_backend::DbBackend;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Bytes sent per request when uploading or downloading: a request's line
/// stays well under the server's limit, base64 and all.
pub const CHUNK_BYTES: usize = 4 << 20;

/// How long an upload may go without a chunk before it's taken as
/// abandoned, and collected.
const ABANDONED_UPLOAD: Duration = Duration::from_secs(24 * 3600);

/// The directory of artifacts, on the side that has the database.
pub struct ArtifactStore {
    dir: PathBuf,
}

/// Held while an upload's contents are committed and its row written, so
/// that garbage collection never sees the contents without the row.
pub struct CommitGuard {
    _lock: File,
}

impl ArtifactStore {
    /// The store in `dir`, made when first written to.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Where artifacts go by default: next to the database, in a
    /// directory named after it (`faber.db` keeps them in
    /// `faber-artifacts/`) - one per database, as each collects what its
    /// rows don't name.
    pub fn default_dir(db_path: &Path) -> PathBuf {
        let stem = db_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "faber".to_string());
        db_path.with_file_name(format!("{}-artifacts", stem))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn tmp_dir(&self) -> PathBuf {
        self.dir.join("tmp")
    }

    fn blobs_dir(&self) -> PathBuf {
        self.dir.join("sha256")
    }

    /// The staging file of upload `id`: only ids `begin` makes are taken,
    /// so a caller can't name a path elsewhere.
    fn upload_path(&self, id: &str) -> Result<PathBuf, Box<dyn Error>> {
        if id.len() != 32 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("no upload '{}'", id).into());
        }
        Ok(self.tmp_dir().join(id))
    }

    /// Where the contents with `digest` (`sha256:<hex>`) are.
    pub fn blob_path(&self, digest: &str) -> Result<PathBuf, Box<dyn Error>> {
        let hex = digest
            .strip_prefix("sha256:")
            .filter(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
            .ok_or_else(|| format!("bad digest '{}'", digest))?;
        Ok(self.blobs_dir().join(hex))
    }

    /// Starts an upload, returning its id.
    pub fn begin(&self) -> Result<String, Box<dyn Error>> {
        std::fs::create_dir_all(self.tmp_dir())?;
        let id = random_hex(16)?;
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.upload_path(&id)?)?;
        Ok(id)
    }

    /// Adds `data` to the end of upload `id`.
    pub fn append(&self, id: &str, data: &[u8]) -> Result<(), Box<dyn Error>> {
        let mut file = OpenOptions::new()
            .append(true)
            .open(self.upload_path(id)?)
            .map_err(|_| format!("no upload '{}'", id))?;
        file.write_all(data)?;
        Ok(())
    }

    /// Moves upload `id` to where its contents' digest says, returning
    /// the digest and the size. Until the guard is dropped, garbage isn't
    /// collected: write the row naming the digest first.
    pub fn commit(&self, id: &str) -> Result<(String, u64, CommitGuard), Box<dyn Error>> {
        let path = self.upload_path(id)?;
        let mut file = File::open(&path).map_err(|_| format!("no upload '{}'", id))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 16];
        let mut size = 0u64;
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            size += n as u64;
        }
        let hex: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect();
        let digest = format!("sha256:{}", hex);
        let guard = self.lock(rustix::fs::FlockOperation::LockShared)?;
        std::fs::create_dir_all(self.blobs_dir())?;
        let blob = self.blob_path(&digest)?;
        if blob.exists() {
            // The same contents are already there: keep those.
            std::fs::remove_file(&path)?;
        } else {
            let mut permissions = file.metadata()?.permissions();
            permissions.set_readonly(true);
            std::fs::set_permissions(&path, permissions)?;
            std::fs::rename(&path, &blob)?;
        }
        Ok((digest, size, guard))
    }

    /// Up to `len` bytes of the contents with `digest`, from `offset`:
    /// fewer only at the end.
    pub fn read(&self, digest: &str, offset: u64, len: usize) -> Result<Vec<u8>, Box<dyn Error>> {
        let mut file = self.open(digest)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut data = Vec::with_capacity(len.min(CHUNK_BYTES));
        file.take(len as u64).read_to_end(&mut data)?;
        Ok(data)
    }

    pub fn open(&self, digest: &str) -> Result<File, Box<dyn Error>> {
        let path = self.blob_path(digest)?;
        File::open(&path).map_err(|e| format!("artifact contents {}: {}", digest, e).into())
    }

    fn lock(&self, operation: rustix::fs::FlockOperation) -> Result<CommitGuard, Box<dyn Error>> {
        std::fs::create_dir_all(&self.dir)?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.dir.join("lock"))?;
        rustix::fs::flock(&file, operation)?;
        Ok(CommitGuard { _lock: file })
    }

    /// Removes the contents no row names - `referenced` gives the digests
    /// that are, read once nothing can be committed - and the uploads
    /// abandoned. Returns how many files went.
    pub fn collect_garbage(
        &self,
        referenced: impl FnOnce() -> Result<HashSet<String>, Box<dyn Error>>,
    ) -> Result<usize, Box<dyn Error>> {
        if !self.dir.is_dir() {
            return Ok(0);
        }
        let _guard = self.lock(rustix::fs::FlockOperation::LockExclusive)?;
        let referenced = referenced()?;
        let mut removed = 0;
        for entry in read_dir_if_any(&self.blobs_dir())? {
            let entry = entry?;
            let digest = format!("sha256:{}", entry.file_name().to_string_lossy());
            if !referenced.contains(&digest) {
                std::fs::remove_file(entry.path())?;
                removed += 1;
            }
        }
        let now = SystemTime::now();
        for entry in read_dir_if_any(&self.tmp_dir())? {
            let entry = entry?;
            let modified = entry.metadata()?.modified()?;
            if now.duration_since(modified).unwrap_or_default() > ABANDONED_UPLOAD {
                std::fs::remove_file(entry.path())?;
                removed += 1;
            }
        }
        Ok(removed)
    }
}

fn read_dir_if_any(dir: &Path) -> Result<Vec<std::io::Result<std::fs::DirEntry>>, Box<dyn Error>> {
    match std::fs::read_dir(dir) {
        Ok(entries) => Ok(entries.collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e.into()),
    }
}

fn random_hex(bytes: usize) -> Result<String, Box<dyn Error>> {
    let mut buf = vec![0u8; bytes];
    File::open("/dev/urandom")?.read_exact(&mut buf)?;
    Ok(buf.iter().map(|b| format!("{:02x}", b)).collect())
}

/// Ends upload `upload` into `store`, recording it as `artifact` in the
/// database `conn` locks: what `DbBackend::artifact_upload_finish` does
/// where the database is. The store is locked before the database, as
/// garbage collection does.
pub fn finish_upload<C: std::ops::Deref<Target = rusqlite::Connection>>(
    store: &ArtifactStore,
    upload: &str,
    artifact: &NewArtifact,
    conn: impl FnOnce() -> Result<C, Box<dyn Error>>,
) -> Result<crate::db::ArtifactRow, Box<dyn Error>> {
    check_name(&artifact.name)?;
    let (digest, size, _guard) = store.commit(upload)?;
    let conn = conn()?;
    crate::db::add_artifact(&conn, artifact, &digest, size)
}

/// See `DbBackend::read_artifact`.
pub fn read_artifact(
    conn: &rusqlite::Connection,
    store: &ArtifactStore,
    id: i64,
    offset: u64,
    len: usize,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let artifact =
        crate::db::get_artifact(conn, id)?.ok_or_else(|| format!("no artifact {}", id))?;
    store.read(&artifact.digest, offset, len.min(CHUNK_BYTES))
}

/// See `DbBackend::gc_artifacts`.
pub fn collect_garbage<C: std::ops::Deref<Target = rusqlite::Connection>>(
    store: &ArtifactStore,
    conn: impl FnOnce() -> Result<C, Box<dyn Error>>,
) -> Result<usize, Box<dyn Error>> {
    store.collect_garbage(|| {
        let conn = conn()?;
        crate::db::artifact_digests(&conn)
    })
}

/// What an artifact is to be saved as.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct NewArtifact {
    pub task_id: i64,
    pub name: String,
    pub media_type: String,
    /// The agent that saved it, if any.
    pub agent: Option<String>,
}

/// Checks an artifact's name: what it's downloaded as, so a file name,
/// not a path.
pub fn check_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.len() > 255
        || name.contains(['/', '\\', '\0'])
        || name.chars().any(char::is_control)
    {
        return Err(format!(
            "'{}' can't be an artifact's name: give a file name, without '/'",
            name
        ));
    }
    Ok(())
}

/// Saves everything `reader` gives as an artifact, through `db`: a chunk
/// at a time, whether the database is here or on a server.
pub fn upload(
    db: &dyn DbBackend,
    reader: &mut dyn Read,
    artifact: &NewArtifact,
) -> Result<crate::db::ArtifactRow, Box<dyn Error>> {
    check_name(&artifact.name)?;
    let upload = db.artifact_upload_begin()?;
    let mut buf = vec![0u8; CHUNK_BYTES];
    loop {
        let mut filled = 0;
        while filled < buf.len() {
            let n = reader.read(&mut buf[filled..])?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        if filled > 0 {
            db.artifact_upload_append(&upload, &buf[..filled])?;
        }
        if filled < buf.len() {
            break;
        }
    }
    db.artifact_upload_finish(&upload, artifact)
}

/// Writes the contents of artifact `id` to `writer`, through `db`, a
/// chunk at a time. Returns how many bytes.
pub fn download(
    db: &dyn DbBackend,
    id: i64,
    writer: &mut dyn Write,
) -> Result<u64, Box<dyn Error>> {
    let mut offset = 0u64;
    loop {
        let data = db.read_artifact(id, offset, CHUNK_BYTES)?;
        writer.write_all(&data)?;
        offset += data.len() as u64;
        if data.len() < CHUNK_BYTES {
            break;
        }
    }
    writer.flush()?;
    Ok(offset)
}

/// A media type for a file named `name`, from its extension; for anything
/// else, generic bytes.
pub fn guess_media_type(name: &str) -> &'static str {
    let extension = name
        .rsplit_once('.')
        .map(|(_, e)| e.to_ascii_lowercase())
        .unwrap_or_default();
    match extension.as_str() {
        "txt" | "log" => "text/plain",
        "md" | "markdown" => "text/markdown",
        "csv" => "text/csv",
        "html" | "htm" => "text/html",
        "css" => "text/css",
        "js" | "mjs" => "text/javascript",
        "json" => "application/json",
        "xml" => "application/xml",
        "yaml" | "yml" => "application/yaml",
        "toml" => "application/toml",
        "pdf" => "application/pdf",
        "zip" => "application/zip",
        "gz" | "tgz" => "application/gzip",
        "tar" => "application/x-tar",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        "webp" => "image/webp",
        "diff" | "patch" => "text/x-diff",
        "rs" | "py" | "c" | "h" | "go" | "sh" | "java" | "ts" => "text/plain",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (ArtifactStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!("faber-artifacts-{}", random_hex(8).unwrap()));
        (ArtifactStore::new(&dir), dir)
    }

    #[test]
    fn test_contents_are_kept_by_digest_once() {
        let (store, dir) = store();
        let a = store.begin().unwrap();
        store.append(&a, b"hello ").unwrap();
        store.append(&a, b"world").unwrap();
        let (digest, size, guard) = store.commit(&a).unwrap();
        drop(guard);
        assert_eq!(
            digest,
            "sha256:b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
        assert_eq!(size, 11);
        assert_eq!(store.read(&digest, 6, 100).unwrap(), b"world");
        assert!(
            store.append(&a, b"more").is_err(),
            "a committed upload is gone"
        );

        let b = store.begin().unwrap();
        store.append(&b, b"hello world").unwrap();
        assert_eq!(store.commit(&b).unwrap().0, digest);
        assert_eq!(std::fs::read_dir(dir.join("sha256")).unwrap().count(), 1);
        assert_eq!(std::fs::read_dir(dir.join("tmp")).unwrap().count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_only_its_own_names_are_taken() {
        let (store, _) = store();
        assert!(store.append("../../etc/passwd", b"x").is_err());
        assert!(store.read("sha256:../x", 0, 1).is_err());
        assert!(store.blob_path("md5:00").is_err());
        assert!(check_name("report.pdf").is_ok());
        for bad in ["", "..", "a/b", "a\\b", "a\nb"] {
            assert!(check_name(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn test_garbage_is_what_no_row_names() {
        let (store, dir) = store();
        let mut digests = Vec::new();
        for content in [&b"keep"[..], b"drop"] {
            let id = store.begin().unwrap();
            store.append(&id, content).unwrap();
            digests.push(store.commit(&id).unwrap().0);
        }
        let pending = store.begin().unwrap();
        let keep = digests[0].clone();
        let removed = store.collect_garbage(|| Ok(HashSet::from([keep]))).unwrap();
        assert_eq!(removed, 1);
        assert!(store.open(&digests[0]).is_ok());
        assert!(store.open(&digests[1]).is_err());
        // An upload under way stays.
        store.append(&pending, b"x").unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_default_dir_is_named_after_the_database() {
        assert_eq!(
            ArtifactStore::default_dir(Path::new("/var/lib/faber/faber.db")),
            PathBuf::from("/var/lib/faber/faber-artifacts")
        );
    }
}
