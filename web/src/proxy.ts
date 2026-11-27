import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import {
  AuthType,
  SERVER_SIDE_ONLY__PAID_ENTERPRISE_FEATURES_ENABLED,
  SERVER_SIDE_ONLY__AUTH_TYPE,
} from "./lib/constants";
import { buildContentSecurityPolicy } from "./lib/contentSecurityPolicy";

// Authentication cookie names (matches backend constants)
const FASTAPI_USERS_AUTH_COOKIE_NAME = "fastapiusersauth";
const ANONYMOUS_USER_COOKIE_NAME = "onyx_anonymous_user";

// Protected route prefixes (require authentication)
const PROTECTED_ROUTES = ["/app", "/admin", "/agents", "/connector"];

// Public route prefixes (no authentication required)
const PUBLIC_ROUTES = ["/auth", "/anonymous", "/_next", "/api"];

// NOTE: have to have the "/:path*" here since NextJS doesn't allow any real JS to
// be run before the config is defined e.g. if we try and do a .map it will complain
export const config = {
  matcher: [
    // Every rendered page, so each response gets a per-request CSP nonce.
    // Auth checks and /ee rewriting below are limited to their own routes.
    "/((?!api|_next/static|_next/image|ph_ingest|favicon.ico).*)",
  ],
};

// Enterprise Edition specific routes (ONLY these get /ee rewriting)
const EE_ROUTES = [
  "/admin/groups",
  "/admin/performance/usage",
  "/admin/performance/query-history",
  "/admin/theme",
  "/admin/performance/custom-analytics",
  "/admin/standard-answer",
  "/agents/stats",
];

function isUnderRoute(pathname: string, route: string): boolean {
  return pathname === route || pathname.startsWith(`${route}/`);
}

export async function proxy(request: NextRequest) {
  const pathname = request.nextUrl.pathname;

  // Auth Check: Fast-fail at edge if no cookie (defense in depth)
  // Note: Layouts still do full verification (token validity, roles, etc.)
  const isProtectedRoute = PROTECTED_ROUTES.some((route) =>
    isUnderRoute(pathname, route)
  );
  const isPublicRoute = PUBLIC_ROUTES.some((route) =>
    pathname.startsWith(route)
  );

  if (isProtectedRoute && !isPublicRoute) {
    const authCookie = request.cookies.get(FASTAPI_USERS_AUTH_COOKIE_NAME);
    const anonymousCookie = request.cookies.get(ANONYMOUS_USER_COOKIE_NAME);

    // Allow access if user has either a regular auth cookie or anonymous user cookie
    if (!authCookie && !anonymousCookie) {
      const loginUrl = new URL("/auth/login", request.url);
      // Preserve full URL including query params and hash for deep linking
      const fullPath = pathname + request.nextUrl.search + request.nextUrl.hash;
      loginUrl.searchParams.set("next", fullPath);
      return NextResponse.redirect(loginUrl);
    }
  }

  // Next.js reads the nonce from the request's CSP header and adds it to the
  // scripts it renders; the root layout reads x-nonce for its own scripts.
  const nonce = btoa(crypto.randomUUID());
  const contentSecurityPolicy = buildContentSecurityPolicy(nonce);
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);
  requestHeaders.set("Content-Security-Policy", contentSecurityPolicy);

  let response: NextResponse;
  // Enterprise Edition: Rewrite EE-specific routes to /ee prefix
  if (
    SERVER_SIDE_ONLY__PAID_ENTERPRISE_FEATURES_ENABLED &&
    EE_ROUTES.some((route) => pathname.startsWith(route))
  ) {
    const newUrl = new URL(`/ee${pathname}`, request.url);
    response = NextResponse.rewrite(newUrl, {
      request: { headers: requestHeaders },
    });
  } else {
    response = NextResponse.next({ request: { headers: requestHeaders } });
  }

  response.headers.set("Content-Security-Policy", contentSecurityPolicy);
  return response;
}
