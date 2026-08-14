#!/usr/bin/env node
/**
 * Generates/refreshes every publishable workspace package's jsr.json from
 * its package.json (issue #25: dual npm+JSR distribution) — a single
 * source of truth instead of two hand-maintained, driftable manifests.
 *
 * JSR package names must be scoped; this repo's npm packages are unscoped
 * `mallory-*` (the `@mallory` npm scope belongs to someone else — see
 * docs/RELEASING.md), so the JSR name is the same string under `@johnhenry`.
 *
 * JSR publishes/type-checks the TypeScript SOURCE directly (not the
 * compiled `dist/` npm ships), so `exports` points at `./src/index.ts`.
 * Bare-specifier imports of workspace siblings ("mallory-tensor-core") and
 * external npm deps ("apache-arrow") need an explicit import map for JSR's
 * resolver — plain npm `dependencies`/`peerDependencies` entries in
 * package.json are translated into `imports` here: a `mallory-*` workspace
 * dependency maps to `jsr:@johnhenry/<name>@<version>`, anything else maps
 * to `npm:<name>@<version>`.
 *
 * Run manually after adding/bumping a dependency, or wire into a
 * pre-publish CI step (see .github/workflows/release.yml's jsr job).
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;

const PACKAGE_DIRS = [
  "packages/fft",
  "packages/data",
  "packages/frame-arrow",
  "packages/frame-parquet",
  "packages/image",
  "packages/mcp",
  "packages/scalar-types",
  "packages/signal",
  "packages/telemetry",
  "packages/tensor-autograd",
  "packages/tensor-compile",
  "packages/tensor-core",
  "packages/tensor-wasm",
  "packages/tensor-webgpu",
  "adapters/adapter-math",
  "adapters/adapter-onnx",
  "scalars/unit",
];

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function buildImports(pkg) {
  const deps = { ...pkg.dependencies, ...pkg.peerDependencies };
  const imports = {};
  for (const [name, version] of Object.entries(deps)) {
    if (name.startsWith("mallory-")) {
      // Workspace sibling, also published to JSR under the same @johnhenry scope.
      const range = version.replace(/^[\^~]/, "");
      imports[name] = `jsr:@johnhenry/${name}@^${range}`;
    } else {
      imports[name] = `npm:${name}@${version}`;
    }
  }
  return imports;
}

for (const dir of PACKAGE_DIRS) {
  const pkgPath = join(ROOT, dir, "package.json");
  const pkg = readJson(pkgPath);
  const imports = buildImports(pkg);

  const jsrConfig = {
    name: `@johnhenry/${pkg.name}`,
    version: pkg.version,
    exports: "./src/index.ts",
    ...(Object.keys(imports).length > 0 ? { imports } : {}),
  };

  const jsrPath = join(ROOT, dir, "jsr.json");
  writeFileSync(jsrPath, `${JSON.stringify(jsrConfig, null, 2)}\n`);
  console.log(`wrote ${dir}/jsr.json (@johnhenry/${pkg.name}@${pkg.version})`);
}
