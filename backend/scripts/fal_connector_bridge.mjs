#!/usr/bin/env node
/**
 * Minimal authenticated bridge for the Python API server.
 *
 * Credentials stay inside the Replit fal.ai connector. This script accepts one
 * JSON request on stdin and returns a JSON-safe response envelope on stdout.
 */
import connectorSdk from "@replit/connectors-sdk";

const { ReplitConnectors } = connectorSdk;

const readInput = async () => {
  let raw = "";
  for await (const chunk of process.stdin) raw += chunk;
  if (!raw.trim()) throw new Error("Missing bridge request.");
  return JSON.parse(raw);
};

try {
  const input = await readInput();
  const proxyFetch = new ReplitConnectors().createProxyFetch(
    input.connectorName || "falai",
  );
  const response = await proxyFetch(input.url, {
    method: input.method || "GET",
    headers: {
      Accept: "application/json",
      ...(input.payload ? { "Content-Type": "application/json" } : {}),
    },
    body: input.payload ? JSON.stringify(input.payload) : undefined,
  });
  const body = await response.text();
  process.stdout.write(JSON.stringify({
    ok: true,
    status: response.status,
    body,
  }));
} catch (error) {
  process.stdout.write(JSON.stringify({
    ok: false,
    error: error instanceof Error ? error.message : "The fal.ai connector request failed.",
  }));
  process.exitCode = 1;
}