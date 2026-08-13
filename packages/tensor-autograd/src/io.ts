/**
 * Checkpoint format (issue #42): `writeCheckpoint`/`loadCheckpoint` for the
 * named-tensor dict `Module.stateDict()`/`loadStateDict()` produce/consume.
 *
 * A simple length-prefixed sequence of (name, npy-bytes) pairs, built on
 * `Tensor`'s own `toNpy()`/`fromNpy()` per-tensor serialization -- NOT
 * numpy's real `.npz` format (which is a zip archive of `.npy` files).
 * Adding a zip dependency was out of scope for v1; this is a
 * mallory-plus-specific container, not numpy-`.npz`-compatible. A future
 * version could add real `.npz` I/O as a separate function if Python
 * interop for checkpoints becomes a concrete need (mirroring how
 * `mallory-interop`'s `read_ipc`/`read_parquet` already handle the
 * dataframe side of Python interop).
 *
 * Layout: `"MPCK"` magic (4 bytes) + version (1 byte, currently 1) + entry
 * count (u32 LE) + that many `[nameByteLen: u32 LE][npyByteLen: u32 LE]
 * [name utf8 bytes][npy bytes]` entries, back to back.
 */
import { Tensor } from "mallory-tensor-core";

const MAGIC = [0x4d, 0x50, 0x43, 0x4b]; // "MPCK"
const VERSION = 1;
const HEADER_LEN = 9; // 4 magic + 1 version + 4 count

/** Serialize a named-tensor dict (e.g. from `Module.stateDict()`) to a checkpoint byte buffer. */
export function writeCheckpoint(stateDict: Readonly<Record<string, Tensor>>): Uint8Array {
  const entries = Object.entries(stateDict);
  const encoder = new TextEncoder();
  const parts: Uint8Array[] = [];

  const header = new Uint8Array(HEADER_LEN);
  header.set(MAGIC, 0);
  header[4] = VERSION;
  new DataView(header.buffer).setUint32(5, entries.length, true);
  parts.push(header);

  for (const [name, tensor] of entries) {
    const nameBytes = encoder.encode(name);
    const npyBytes = tensor.toNpy();
    const lenPrefix = new Uint8Array(8);
    const lenView = new DataView(lenPrefix.buffer);
    lenView.setUint32(0, nameBytes.length, true);
    lenView.setUint32(4, npyBytes.length, true);
    parts.push(lenPrefix, nameBytes, npyBytes);
  }

  const total = parts.reduce((sum, p) => sum + p.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

/** Deserialize a checkpoint byte buffer back into a named-tensor dict (e.g. for `Module.loadStateDict()`). Throws a clear error on malformed/truncated/unsupported-version input, rather than silently returning partial or wrong data. */
export function loadCheckpoint(bytes: Uint8Array): Record<string, Tensor> {
  const decoder = new TextDecoder();
  if (bytes.length < HEADER_LEN) {
    throw new Error(`loadCheckpoint: truncated checkpoint (${bytes.length} bytes, need at least ${HEADER_LEN} for the header)`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < MAGIC.length; i++) {
    if (view.getUint8(i) !== MAGIC[i]) {
      throw new Error("loadCheckpoint: not a mallory-plus checkpoint (bad magic bytes)");
    }
  }
  const version = view.getUint8(4);
  if (version !== VERSION) {
    throw new Error(`loadCheckpoint: unsupported checkpoint version ${version} (this build supports version ${VERSION})`);
  }
  const count = view.getUint32(5, true);

  const out: Record<string, Tensor> = {};
  let offset = HEADER_LEN;
  for (let i = 0; i < count; i++) {
    if (offset + 8 > bytes.length) throw new Error(`loadCheckpoint: truncated checkpoint (entry ${i}'s length prefix)`);
    const nameLen = view.getUint32(offset, true);
    const npyLen = view.getUint32(offset + 4, true);
    offset += 8;
    if (offset + nameLen + npyLen > bytes.length) {
      throw new Error(`loadCheckpoint: truncated checkpoint (entry ${i}'s data)`);
    }
    const name = decoder.decode(bytes.subarray(offset, offset + nameLen));
    offset += nameLen;
    const npyBytes = bytes.subarray(offset, offset + npyLen);
    offset += npyLen;
    if (name in out) throw new Error(`loadCheckpoint: duplicate parameter name "${name}" in checkpoint`);
    out[name] = Tensor.fromNpy(npyBytes);
  }
  return out;
}
