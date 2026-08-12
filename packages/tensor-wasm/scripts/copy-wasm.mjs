import { copyFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = join(
  here,
  "../../../target/wasm32-unknown-unknown/release/tensor_wasm_kernels.wasm",
);
// cargo always writes to the same path regardless of --features, so the
// scalar and SIMD128 builds (issue #13) must each be copied out to a
// DIFFERENT destination filename immediately after building, before the
// other variant's build overwrites the source.
const destinationName = process.argv[2] ?? "tensor_wasm_kernels.wasm";
const destination = join(here, "../wasm", destinationName);

await mkdir(dirname(destination), { recursive: true });
await copyFile(source, destination);
console.log(`copied ${source} -> ${destination}`);
