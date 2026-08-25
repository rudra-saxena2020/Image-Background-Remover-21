const configuredApiUrl = (import.meta.env.VITE_API_URL as string | undefined)?.trim();
const apiBaseUrl = configuredApiUrl ? configuredApiUrl.replace(/\/+$/, "") : "";

export function apiUrl(input: string): string {
  if (!apiBaseUrl || !input.startsWith("/")) return input;
  return `${apiBaseUrl}${input}`;
}

export function apiFetch(input: string, init?: RequestInit): Promise<Response> {
  return fetch(apiUrl(input), init);
}