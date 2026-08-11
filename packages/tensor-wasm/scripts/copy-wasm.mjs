import { copyFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = join(
  here,
  "../../../target/wasm32-unknown-unknown/release/tensor_wasm_kernels.wasm",
);
const destination = join(here, "../wasm/tensor_wasm_kernels.wasm");

await mkdir(dirname(destination), { recursive: true });
await copyFile(source, destination);
console.log(`copied ${source} -> ${destination}`);
