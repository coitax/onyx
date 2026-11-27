/**
 * Citation and search clicks must open a stored document link only when its
 * scheme is http, https, or mailto. window.open is called with the stored
 * string today, so these tests fail until that call is restricted.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { Citation } from "@/components/search/results/Citation";
import SearchCard from "@/ee/sections/SearchCard";
import { openDocument } from "@/lib/search/utils";
import {
  LoadedOnyxDocument,
  OnyxDocument,
  SearchDocWithContent,
} from "@/lib/search/interfaces";
import { ValidSources } from "@/lib/types";
import { MemoizedLink } from "@/app/app/message/MemoizedTextComponents";
import { InternalSearchToolRenderer } from "@/app/app/message/messageComponents/timeline/renderers/search/InternalSearchToolRenderer";
import { FetchToolRenderer } from "@/app/app/message/messageComponents/timeline/renderers/fetch/FetchToolRenderer";
import { RenderType } from "@/app/app/message/messageComponents/interfaces";
import { PacketType } from "@/app/app/services/streamingModels";

const UNSAFE_LINKS = [
  "javascript:void(0)",
  "JavaScript:void(0)",
  " javascript:void(0)",
  "\tjavascript:void(0)",
  "\njavascript:void(0)",
  "java\tscript:void(0)",
  "java\nscript:void(0)",
  "\u0000javascript:void(0)",
  "data:text/html,hi",
  "vbscript:msgbox(1)",
  "file:///tmp/x",
];

const SAFE_HTTPS = "https://example.com/policy";
const SAFE_MAILTO = "mailto:person@example.com";

function baseDocument(link: string, title = "notes.txt"): OnyxDocument {
  return {
    document_id: "doc-1",
    semantic_identifier: title,
    link,
    source_type: ValidSources.File,
    blurb: "Company policy text",
    boost: 0,
    hidden: false,
    score: 0,
    chunk_ind: 0,
    match_highlights: [],
    metadata: {},
    updated_at: null,
    is_internet: false,
  };
}

function documentThatFlipsOnUse(title = "notes.txt"): {
  document: OnyxDocument;
  arm: () => void;
} {
  const document = baseDocument(SAFE_HTTPS, title);
  let armed = false;
  let reads = 0;
  Object.defineProperty(document, "link", {
    configurable: true,
    get() {
      if (!armed) return SAFE_HTTPS;
      reads += 1;
      // The first read after arming is the value a check would see.
      // The next read is what a second lookup would open.
      return reads === 1 ? SAFE_HTTPS : "javascript:void(0)";
    },
  });
  return {
    document,
    arm: () => {
      armed = true;
    },
  };
}

function openedUrls(): string[] {
  return ((window.open as jest.Mock).mock.calls as unknown[][]).map((call) =>
    String(call[0])
  );
}

function assertOpenedUrlsAreSafe(): void {
  for (const url of openedUrls()) {
    let protocol = "";
    try {
      protocol = new URL(url).protocol;
    } catch {
      protocol = "relative";
    }
    expect(["http:", "https:", "mailto:"]).toContain(protocol);
  }
}

beforeEach(() => {
  window.open = jest.fn().mockReturnValue(null);
});

describe("openDocument", () => {
  it("opens http, https, and mailto links", () => {
    openDocument(baseDocument(SAFE_HTTPS));
    openDocument(baseDocument(SAFE_MAILTO));
    expect(openedUrls()).toEqual([SAFE_HTTPS, SAFE_MAILTO]);
  });

  it("does not open a stored script, data, or file link", () => {
    for (const link of UNSAFE_LINKS) {
      (window.open as jest.Mock).mockClear();
      openDocument(baseDocument(link));
      assertOpenedUrlsAreSafe();
    }
  });

  it("does not open a link that changes between the check and the open", () => {
    const { document, arm } = documentThatFlipsOnUse();
    arm();
    openDocument(document);
    assertOpenedUrlsAreSafe();
    expect(openedUrls()).not.toContain("javascript:void(0)");
  });

  it("previews a file that has no link instead of opening a window", () => {
    const presented: OnyxDocument[] = [];
    const document = baseDocument("");
    openDocument(document, (next) => presented.push(next as OnyxDocument));
    expect(window.open).not.toHaveBeenCalled();
    expect(presented).toHaveLength(1);
  });
});

function clickLabel(label: string): void {
  const matches = screen.getAllByText(label);
  const visible =
    matches.find((element) => element.closest("[aria-hidden='true']") == null) ??
    matches[0];
  fireEvent.click(visible);
}

describe("Citation", () => {
  function renderCitation(document: OnyxDocument): void {
    render(
      <Citation
        document_info={{
          document: {
            ...document,
            icon: () => null,
          } as LoadedOnyxDocument,
          updatePresentingDocument: () => undefined,
        }}
      >
        [1]
      </Citation>
    );
  }

  it("does not open an unsafe citation link", () => {
    renderCitation(baseDocument("javascript:void(0)"));
    clickLabel("File");
    assertOpenedUrlsAreSafe();
  });

  it("does not open a citation link that changes between the check and the open", () => {
    const { document, arm } = documentThatFlipsOnUse();
    const loaded = document as LoadedOnyxDocument;
    loaded.icon = () => null;
    render(
      <Citation
        document_info={{
          document: loaded,
          updatePresentingDocument: () => undefined,
        }}
      >
        [1]
      </Citation>
    );
    arm();
    clickLabel("File");
    assertOpenedUrlsAreSafe();
  });
});

describe("MemoizedLink", () => {
  function renderLink(document: OnyxDocument): void {
    render(
      <MemoizedLink
        document={document}
        updatePresentingDocument={() => undefined}
      >
        [1]
      </MemoizedLink>
    );
  }

  it("does not open an unsafe source link", () => {
    renderLink(baseDocument("javascript:void(0)", "policy.txt"));
    clickLabel("File");
    assertOpenedUrlsAreSafe();
  });

  it("does not open a source link that changes between the check and the open", () => {
    const { document, arm } = documentThatFlipsOnUse("policy.txt");
    renderLink(document);
    arm();
    clickLabel("File");
    assertOpenedUrlsAreSafe();
  });
});

describe("SearchCard", () => {
  function searchDoc(document: OnyxDocument): SearchDocWithContent {
    return {
      document_id: document.document_id,
      chunk_ind: document.chunk_ind,
      semantic_identifier: document.semantic_identifier || "notes.txt",
      link: document.link,
      blurb: document.blurb,
      source_type: document.source_type,
      boost: document.boost,
      hidden: document.hidden,
      metadata: document.metadata,
      score: document.score,
      match_highlights: document.match_highlights,
      updated_at: document.updated_at,
      is_internet: false,
    };
  }

  it("does not open an unsafe result link", () => {
    const document = baseDocument("javascript:void(0)", "search-result");
    render(
      <SearchCard document={searchDoc(document)} onDocumentClick={() => undefined} />
    );
    clickLabel("search-result");
    assertOpenedUrlsAreSafe();
  });

  it("opens an https result link", () => {
    const document = baseDocument(SAFE_HTTPS, "safe-result");
    render(
      <SearchCard document={searchDoc(document)} onDocumentClick={() => undefined} />
    );
    clickLabel("safe-result");
    expect(openedUrls()).toEqual([SAFE_HTTPS]);
  });

  it("does not open a result link that changes between the check and the open", () => {
    const document = searchDoc(baseDocument(SAFE_HTTPS, "flip-result"));
    let armed = false;
    let reads = 0;
    Object.defineProperty(document, "link", {
      configurable: true,
      get() {
        if (!armed) return SAFE_HTTPS;
        reads += 1;
        return reads === 1 ? SAFE_HTTPS : "javascript:void(0)";
      },
    });
    render(
      <SearchCard document={document} onDocumentClick={() => undefined} />
    );
    armed = true;
    clickLabel("flip-result");
    assertOpenedUrlsAreSafe();
  });
});

const placement = { turn_index: 0 };

function searchPackets(document: OnyxDocument) {
  return [
    {
      placement,
      obj: { type: PacketType.SEARCH_TOOL_START as const },
    },
    {
      placement,
      obj: {
        type: PacketType.SEARCH_TOOL_QUERIES_DELTA as const,
        queries: ["policy"],
      },
    },
    {
      placement,
      obj: {
        type: PacketType.SEARCH_TOOL_DOCUMENTS_DELTA as const,
        documents: [document],
      },
    },
    {
      placement,
      obj: { type: PacketType.SECTION_END as const },
    },
  ];
}

function renderSearch(document: OnyxDocument, renderType: RenderType): void {
  render(
    <InternalSearchToolRenderer
      packets={searchPackets(document)}
      state={{}}
      onComplete={() => undefined}
      renderType={renderType}
      animate={false}
      stopPacketSeen
      children={(steps) => (
        <>
          {steps.map((step, index) => (
            <div key={index}>{step.content}</div>
          ))}
        </>
      )}
    />
  );
}

describe("InternalSearchToolRenderer", () => {
  const modes = [RenderType.HIGHLIGHT, RenderType.INLINE, RenderType.FULL];

  it.each(modes)("does not open an unsafe result in %s mode", (renderType) => {
    renderSearch(baseDocument("javascript:void(0)", `search-${renderType}`), renderType);
    clickLabel(`search-${renderType}`);
    assertOpenedUrlsAreSafe();
  });

  it.each(modes)(
    "does not open a result that changes between the check and the open in %s mode",
    (renderType) => {
      const { document, arm } = documentThatFlipsOnUse(`flip-${renderType}`);
      renderSearch(document, renderType);
      arm();
      clickLabel(`flip-${renderType}`);
      assertOpenedUrlsAreSafe();
    }
  );
});

function renderFetch(
  renderType: RenderType,
  documents: OnyxDocument[],
  urls: string[]
): void {
  const packets = [
    { placement, obj: { type: PacketType.FETCH_TOOL_START as const } },
    { placement, obj: { type: PacketType.FETCH_TOOL_URLS as const, urls } },
    {
      placement,
      obj: { type: PacketType.FETCH_TOOL_DOCUMENTS as const, documents },
    },
    { placement, obj: { type: PacketType.SECTION_END as const } },
  ];
  if (documents.length === 0) {
    packets.splice(2, 1);
  }
  render(
    <FetchToolRenderer
      packets={packets}
      state={{}}
      onComplete={() => undefined}
      renderType={renderType}
      animate={false}
      stopPacketSeen
      children={(steps) => (
        <>
          {steps.map((step, index) => (
            <div key={index}>{step.content}</div>
          ))}
        </>
      )}
    />
  );
}

describe("FetchToolRenderer", () => {
  const modes = [RenderType.HIGHLIGHT, RenderType.FULL];

  it.each(modes)("does not open an unsafe document link in %s mode", (renderType) => {
    renderFetch(
      renderType,
      [baseDocument("javascript:void(0)", `fetch-${renderType}`)],
      []
    );
    clickLabel(`fetch-${renderType}`);
    assertOpenedUrlsAreSafe();
  });

  it.each(modes)("does not open an unsafe fetched url in %s mode", (renderType) => {
    const url = "javascript:void(0)";
    renderFetch(renderType, [], [url]);
    clickLabel(url);
    assertOpenedUrlsAreSafe();
  });

  it.each(modes)(
    "does not open a document link that changes between the check and the open in %s mode",
    (renderType) => {
      const { document, arm } = documentThatFlipsOnUse(
        `fetch-flip-${renderType}`
      );
      renderFetch(renderType, [document], []);
      arm();
      clickLabel(`fetch-flip-${renderType}`);
      assertOpenedUrlsAreSafe();
    }
  );
});
