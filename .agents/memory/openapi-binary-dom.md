---
name: OpenAPI binary format and Node.js DOM types
description: Adding format:binary to OpenAPI request body causes TypeScript errors in the api-zod lib
---

**Rule:** Never use `format: binary` in OpenAPI request body schemas for this workspace.

**Why:** Orval generates `zod.instanceof(File)` for binary fields. The `File` type is a DOM API not present in the `es2022` TypeScript lib used by `lib/api-zod`. This causes `error TS2304: Cannot find name 'File'` during `typecheck:libs`.

**Workarounds:**
1. Remove `format: binary` from request body schemas (cleanest — Python FastAPI handles multipart natively without the schema)
2. Add `"dom"` to `lib/api-zod/tsconfig.json` `lib` array (done for this project)

**How to apply:** For multipart/form-data endpoints returning binary: omit binary fields from the OpenAPI request body schema. The Python backend validates the upload directly; the Zod schemas are not used server-side for binary.
