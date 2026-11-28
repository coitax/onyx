import { render, waitFor } from "@tests/setup/test-utils";
import JSZip from "jszip";
import { PreviewContext } from "@/sections/modals/PreviewModal/interfaces";
import PreviewModal from "@/sections/modals/PreviewModal/PreviewModal";
import { docxVariant } from "@/sections/modals/PreviewModal/variants/docxVariant";
import { resolveVariant } from "@/sections/modals/PreviewModal/variants";

/**
 * A DOCX preview must not place an embedded HTML altChunk into an iframe that
 * can run script as the Onyx page. docx-preview does that unless
 * renderAltChunks is turned off or the iframe is sandboxed off the page origin.
 *
 * These tests fail while that rendering is still enabled. They pass once the
 * preview no longer gives the embedded HTML the page origin.
 *
 * The bytes are the original package. Chat upload keeps those bytes whenever
 * the multipart part's content type is not the Word MIME; the preview then
 * downloads them and selects this renderer because the file name ends in .docx.
 * Every preview entry point (chat attachments, shared chats, the app page,
 * the NRF page, and expandable file content) uses PreviewModal, which uses
 * docxVariant. The craft file browser does not; it shows text and is not covered.
 */

const PROBE = "ALTCHUNK_PROBE_TOKEN";

const ALT_CHUNK_HTML = `<!DOCTYPE html><html><body><script>window.parent.__altChunkProbe=${JSON.stringify(
  PROBE
)}</script></body></html>`;

function previewContext(fileUrl: string): PreviewContext {
  return {
    fileContent: "",
    fileUrl,
    fileName: "note.docx",
    language: "plaintext",
    lineCount: 0,
    fileSize: "",
    zoom: 100,
    onZoomIn: () => undefined,
    onZoomOut: () => undefined,
  };
}

async function buildAltChunkDocx(): Promise<Uint8Array> {
  const zip = new JSZip();
  zip.file(
    "[Content_Types].xml",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="html" ContentType="text/html"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>`
  );
  zip.file(
    "_rels/.rels",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>`
  );
  zip.file(
    "word/document.xml",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    <w:p><w:r><w:t>visible</w:t></w:r></w:p>
    <w:altChunk r:id="rIdChunk"/>
  </w:body>
</w:document>`
  );
  zip.file(
    "word/_rels/document.xml.rels",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdChunk" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/aFChunk" Target="chunk.html"/>
</Relationships>`
  );
  zip.file("word/chunk.html", ALT_CHUNK_HTML);
  return zip.generateAsync({ type: "uint8array" });
}

function framesThatRunAsPage(root: ParentNode): HTMLIFrameElement[] {
  return Array.from(root.querySelectorAll("iframe")).filter((frame) => {
    const srcdoc = frame.srcdoc || "";
    if (!srcdoc.includes(PROBE)) {
      return false;
    }
    if (!frame.hasAttribute("sandbox")) {
      return true;
    }
    return (
      frame.sandbox.contains("allow-scripts") &&
      frame.sandbox.contains("allow-same-origin")
    );
  });
}

function installDocxFetch(bytes: Uint8Array, contentType: string) {
  // jsdom does not implement blob URLs. Keep the blob PreviewModal created
  // from the download and serve those same bytes back to the preview fetch.
  const blobs = new Map<string, Blob>();
  window.URL.createObjectURL = ((blob: Blob) => {
    const url = `blob:note-docx-${blobs.size}`;
    blobs.set(url, blob);
    return url;
  }) as typeof URL.createObjectURL;
  window.URL.revokeObjectURL = (() => undefined) as typeof URL.revokeObjectURL;

  const fetchImpl = async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.startsWith("blob:")) {
      const blob = blobs.get(url);
      if (!blob) {
        throw new Error(`unknown blob ${url}`);
      }
      // The fetch polyfill's Blob has no arrayBuffer(). FileReader is how
      // this jsdom reads the bytes PreviewModal stored from the download.
      const raw = await new Promise<Uint8Array>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () =>
          resolve(new Uint8Array(reader.result as ArrayBuffer));
        reader.onerror = () => reject(reader.error);
        reader.readAsArrayBuffer(blob);
      });
      return new Response(raw);
    }
    if (
      url.includes("/api/chat/file/") ||
      url === "https://preview.invalid/note.docx"
    ) {
      return new Response(bytes.slice().buffer, {
        status: 200,
        headers: { "Content-Type": contentType },
      });
    }
    throw new Error(`unexpected fetch ${url}`);
  };
  global.fetch = fetchImpl as typeof fetch;
  window.fetch = fetchImpl as typeof fetch;
}

describe("DOCX altChunk preview", () => {
  let bytes: Uint8Array;

  beforeAll(async () => {
    bytes = await buildAltChunkDocx();
  });

  beforeEach(async () => {
    // jest.setup loads this polyfill asynchronously and it replaces fetch.
    // Import it first so the test's fetch is the one the preview calls.
    await import("whatwg-fetch");
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("selects the docx renderer for a .docx name even when the stored type is not the Word MIME", () => {
    expect(resolveVariant("note.docx", "text/plain")).toBe(docxVariant);
    expect(resolveVariant("note.docx", "application/octet-stream")).toBe(
      docxVariant
    );
    expect(
      resolveVariant(
        "note.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      )
    ).toBe(docxVariant);
  });

  it("does not run embedded altChunk HTML as the preview page", async () => {
    installDocxFetch(bytes, "application/octet-stream");
    const { container } = render(
      <>
        {docxVariant.renderContent(
          previewContext("https://preview.invalid/note.docx")
        )}
      </>
    );

    await waitFor(() => {
      expect(container).toHaveTextContent("visible");
    });

    expect(framesThatRunAsPage(container)).toHaveLength(0);
  });

  it.each([["application/octet-stream"], ["text/plain"]])(
    "does not run embedded altChunk HTML when PreviewModal downloads a %s docx",
    async (contentType) => {
      installDocxFetch(bytes, contentType);
      render(
        <PreviewModal
          presentingDocument={{
            document_id: "stored-file-id",
            semantic_identifier: "note.docx",
          }}
          onClose={() => undefined}
        />
      );

      await waitFor(() => {
        expect(document.body).toHaveTextContent("visible");
      });

      expect(framesThatRunAsPage(document.body)).toHaveLength(0);
    }
  );
});
