# Releasing

Same deployment substrate as [`johnhenry/mallory`](https://github.com/johnhenry/mallory)
(GitHub Actions + an `NPM_TOKEN` repository secret), driven by
[Changesets](https://github.com/changesets/changesets) because this repo publishes many
independently versioned packages instead of one.

## Prerequisite: the NPM_TOKEN secret

The release workflow is inert until the secret exists:

```bash
gh secret set NPM_TOKEN --repo johnhenry/mallory-plus   # prompts; paste a granular automation token
gh secret list --repo johnhenry/mallory-plus            # verify
```

Mint the token at npmjs.com → Access Tokens → **Granular access token**, with *Read and write*
permission for the `mallory-*` packages (or all packages, since none exist yet). Automation-type
tokens bypass 2FA, which CI requires.

> **⚠️ Adding the secret arms the workflow.** The next push to `main` with no pending changesets
> publishes every non-private workspace package at its current version — today that means
> `mallory-tensor-core`, `mallory-tensor-wasm`, and `mallory-scalar-types` at `0.0.1`. That is
> either a useful way to reserve the names on npm or premature, depending on your intent. To hold
> off, add `"private": true` to a package's `package.json` until it's ready to ship.

## Normal flow

1. **Describe the change** in the PR that makes it:
   ```bash
   npm run changeset      # pick packages, pick bump type, write the summary
   git add .changeset && git commit -m "Add changeset"
   ```
2. **Merge to `main`.** The Release workflow opens (or updates) a **"chore: version packages"** PR
   applying the bumps and changelogs.
3. **Merge the version PR.** The workflow runs again, builds (WASM kernels + TypeScript), tests,
   and publishes every package whose version isn't yet on the registry.

`workflow_dispatch` re-runs the publish step without a new changeset — useful if a publish failed
partway through a multi-package release.

## What CI verifies before publishing

`npm run build` (Rust → `wasm32-unknown-unknown` via cargo + lld, then `tsc` across workspaces) and
`npm test` (unit tests plus the NumPy differential suite — CI installs numpy and asserts it imports,
so those tests can't silently skip; see [TESTING.md](./TESTING.md)).

## Package-name notes

Packages are unscoped `mallory-*` names (the `@mallory` npm scope belongs to another user). The
root package is `private: true` and never publishes. `packages/interop-python` is a PyPI package
outside both the npm and Cargo workspaces, released separately.
