import { AuthType } from "@/lib/constants";
import { getAuthUrlSS } from "@/lib/userSS";

/**
 * Login builds the Google OAuth and OIDC authorize URLs through getAuthUrlSS.
 * An external `next` must not be copied onto that URL. These tests fail until
 * getAuthUrlSS runs validateInternalRedirect before adding `next`.
 */

const MALICIOUS_NEXT_VALUES = [
  "https://evil.example/login",
  "http://evil.example/login",
  "//evil.example/login",
  "/\\evil.example",
  "javascript:alert(1)",
  "/javascript:alert(1)",
  " https://evil.example/login",
  "\\\\evil.example",
  "https://evil.example/login?stolen=1",
];

const SSO_AUTH_TYPES = [AuthType.GOOGLE_OAUTH, AuthType.OIDC, AuthType.CLOUD];

function isSafeNextParam(value: string | null): boolean {
  if (value === null) {
    return true;
  }
  const trimmed = value.trim();
  if (!trimmed.startsWith("/") || trimmed.startsWith("//")) {
    return false;
  }
  if (trimmed.includes("\\")) {
    return false;
  }
  if (trimmed.match(/^[^?#]*:/)) {
    return false;
  }
  return (
    !trimmed.includes("evil.example") &&
    !trimmed.toLowerCase().includes("javascript")
  );
}

describe("getAuthUrlSS does not forward an open redirect", () => {
  beforeEach(() => {
    global.fetch = jest.fn(async () => {
      return {
        ok: true,
        json: async () => ({
          authorization_url: "https://idp.example.com/saml",
        }),
      } as Response;
    });
  });

  it.each(SSO_AUTH_TYPES)(
    "%s omits external and bypass next values",
    async (authType) => {
      for (const nextUrl of MALICIOUS_NEXT_VALUES) {
        const authorizeUrl = await getAuthUrlSS(authType, nextUrl);
        const nextParam = new URL(authorizeUrl).searchParams.get("next");
        expect(isSafeNextParam(nextParam)).toBe(true);
        expect(authorizeUrl).toContain("redirect=true");
      }
    }
  );

  it.each(SSO_AUTH_TYPES)("%s keeps a relative next path", async (authType) => {
    const authorizeUrl = await getAuthUrlSS(authType, "/chat?tab=1");
    expect(new URL(authorizeUrl).searchParams.get("next")).toBe("/chat?tab=1");
    expect(authorizeUrl).toContain("redirect=true");
  });

  it("saml authorize request does not receive an external next", async () => {
    await getAuthUrlSS(AuthType.SAML, "https://evil.example/login");
    expect(global.fetch).toHaveBeenCalled();
    const fetched = String((global.fetch as jest.Mock).mock.calls[0][0]);
    const nextParam = new URL(fetched).searchParams.get("next");
    expect(isSafeNextParam(nextParam)).toBe(true);
  });

  it("saml authorize request keeps a relative next path", async () => {
    await getAuthUrlSS(AuthType.SAML, "/chat");
    const fetched = String((global.fetch as jest.Mock).mock.calls[0][0]);
    expect(new URL(fetched).searchParams.get("next")).toBe("/chat");
  });
});
