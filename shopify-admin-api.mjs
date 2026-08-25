const hostname = process.env.REPLIT_CONNECTORS_HOSTNAME;
const token = process.env.REPL_IDENTITY
  ? `repl ${process.env.REPL_IDENTITY}`
  : process.env.WEB_REPL_RENEWAL
    ? `depl ${process.env.WEB_REPL_RENEWAL}`
    : null;

const [bodyJson, pathArg] = process.argv.slice(2);
const path = pathArg || "/admin/api/2026-04/graphql.json";

if (!hostname || !token) {
  console.error("Missing Replit connector environment.");
  process.exit(1);
}
if (!bodyJson) {
  console.error("Missing GraphQL request body.");
  process.exit(1);
}

const protocol = hostname.startsWith("localhost") ? "http" : "https";
const response = await fetch(`${protocol}://${hostname}/api/v2/proxy${path}`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-Replit-Token": token,
    "Connector-Name": "shopify-store",
  },
  body: bodyJson,
  signal: AbortSignal.timeout(30_000),
});
const text = await response.text();
let parsed;
try {
  parsed = JSON.parse(text);
} catch {
  console.error(text);
  process.exit(1);
}
if (!response.ok || parsed.errors?.length) {
  console.error(JSON.stringify(parsed));
  process.exit(1);
}
console.log(JSON.stringify(parsed));