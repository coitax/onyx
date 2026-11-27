"use client";

import { useEffect, useRef } from "react";
import { useSettingsContext } from "@/providers/SettingsProvider";

interface CustomAnalyticsScriptProps {
  nonce?: string;
}

export default function CustomAnalyticsScript({
  nonce,
}: CustomAnalyticsScriptProps) {
  const { customAnalyticsScript } = useSettingsContext();
  const injectedRef = useRef(false);

  useEffect(() => {
    if (!customAnalyticsScript || injectedRef.current) return;
    injectedRef.current = true;

    const script = document.createElement("script");
    script.type = "text/javascript";
    if (nonce) {
      script.nonce = nonce;
    }
    script.textContent = customAnalyticsScript;
    document.head.appendChild(script);
  }, [customAnalyticsScript, nonce]);

  return null;
}
