# Versioning and Rollback

This repo now uses a simple stable-baseline workflow:

- `VERSION`, `package.json`, and `package-lock.json` hold the current release number.
- Each stable checkpoint gets an annotated git tag in the form `vX.Y.Z`.
- `CHANGELOG.md` records what changed at each stable version.

## Current stable baseline

- Version: `v0.1.0`
- Expected tag: `v0.1.0`
- Base branch: `main`

## Safe rollback workflow

If you start experimenting and want to return to the stable version later, use the tag instead of trying to manually undo changes.

1. Check what changed:

   ```bash
   git status
   git diff --stat
   ```

2. If you want to keep the experimental work, save it first:

   ```bash
   git stash push -u -m "wip before rollback"
   ```

   You can also commit the experiment on its own branch instead.

3. Create a fresh branch from the stable version and run from there:

   ```bash
   git switch -c recovery/v0.1.0 v0.1.0
   ```

This is the safest way to get back to a known-good state without deleting newer work.

## Compare current work against the stable baseline

```bash
git diff v0.1.0..HEAD
git log --oneline --decorate v0.1.0..HEAD
```

## Creating the next stable version

When a future set of changes is working and you want to freeze it:

1. Update:

   ```bash
   VERSION
   package.json
   package-lock.json
   CHANGELOG.md
   ```

2. Verify:

   ```bash
   npm run build
   ```

3. Commit and tag:

   ```bash
   git add VERSION package.json package-lock.json CHANGELOG.md docs/versioning.md README.md
   git commit -m "Release v0.1.1"
   git tag -a v0.1.1 -m "Stable baseline v0.1.1"
   ```

4. Push:

   ```bash
   git push origin main
   git push origin v0.1.1
   ```

## Recommended habit

Use semantic versioning for stable points:

- `0.1.0` for the first stable baseline
- `0.1.1` for safe fixes
- `0.2.0` for meaningful feature additions that still keep the project pre-1.0
