// CommonJS so next.config.js can require it. next.config.js sends the policy
// without a nonce on every response; src/proxy.ts replaces it on rendered pages
// with a per-request nonce that Next.js applies to its own inline scripts.

/**
 * @param {string} [nonce]
 * @returns {string}
 */
function buildContentSecurityPolicy(nonce) {
  const isDev = process.env.NODE_ENV === "development";

  const scriptSrc = ["'self'"];
  if (nonce) {
    scriptSrc.push(`'nonce-${nonce}'`, "'strict-dynamic'");
  }
  if (isDev) {
    // Next.js dev server (React Refresh, source maps) relies on eval.
    scriptSrc.push("'unsafe-eval'");
  }

  const directives = [
    `script-src ${scriptSrc.join(" ")}`,
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
  ];
  if (process.env.NEXT_PUBLIC_CLOUD_ENABLED === "true" && !isDev) {
    directives.push("upgrade-insecure-requests");
  }

  return `${directives.join("; ")};`;
}

module.exports = { buildContentSecurityPolicy };
