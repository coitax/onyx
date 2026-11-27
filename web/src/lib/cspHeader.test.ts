/**
 * The app content security policy has no script-src directive. Without
 * default-src either, a javascript: navigation is not blocked. The policy
 * must set script-src and must not allow unsafe inline scripts.
 */

describe("content security policy", () => {
  it("sets a script-src that does not allow unsafe inline scripts", async () => {
    const nextConfig = require("../../next.config.js");
    const groups = await nextConfig.headers();
    const headers = groups.flatMap(
      (group: { headers: { key: string; value: string }[] }) => group.headers
    );
    const csp = headers.find(
      (header: { key: string }) => header.key === "Content-Security-Policy"
    )?.value as string | undefined;
    expect(csp).toEqual(expect.any(String));
    const scriptSrc = csp
      ?.split(";")
      .map((part) => part.trim())
      .find((part) => part.startsWith("script-src ") || part === "script-src");
    expect(scriptSrc).toBeDefined();
    expect(scriptSrc).not.toMatch(/unsafe-inline/);
    expect(scriptSrc).not.toMatch(/unsafe-eval/);
    expect(scriptSrc).not.toMatch(/\*/);
  });
});
