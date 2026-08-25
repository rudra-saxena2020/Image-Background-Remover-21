---
name: fal.ai connector bridge
description: Compatibility rule for routing backend requests through the attached Replit fal.ai connection.
---

Use the installed connector SDK's `ReplitConnectors` class and its `createProxyFetch()` method, with the active connector slug `falai`; do not assume a named `getConnection` export exists.

**Why:** The available SDK version is CommonJS and exposes the proxy through `ReplitConnectors`. Using a newer or guessed helper fails before a provider request can be sent.

**How to apply:** Keep provider credentials in the connector. Route queue API calls through the bridge, use the full model endpoint when deriving queue fallback URLs, and verify changes with a harmless GET probe rather than billable inference.