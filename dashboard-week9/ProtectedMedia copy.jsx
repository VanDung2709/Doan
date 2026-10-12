import { useEffect, useState } from "react";

// Load protected media with the existing Bearer JWT. Videos fetch only after click.
// Prototype note: video is downloaded as a Blob, not streamed; use small demo clips.
const SERVER = "http://localhost:3001";

export default function ProtectedMedia({ path, token, kind = "image", alt = "Ảnh minh họa", className = "", preload = "none", openOnClick = false }) {
  const authToken = token || localStorage.getItem("token");
  const [enabled, setEnabled] = useState(kind !== "video");
  const [objectUrl, setObjectUrl] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    setEnabled(kind !== "video");
  }, [kind, path]);

  useEffect(() => {
    if (!path || !authToken || !enabled) return undefined;
    let cancelled = false;
    let url = "";
    const controller = new AbortController();
    (async () => {
      try {
        const target = path.startsWith("http") ? path : `${SERVER}${path}`;
        // Accept only same-origin backend media paths; do not send JWT to other hosts.
        const parsed = new URL(target);
        if (parsed.origin !== SERVER || !/^\/(uploads\/videos|uploads\/analyzed|gradcam)\//.test(parsed.pathname)) {
          throw new Error("Đường dẫn media không hợp lệ.");
        }
        const response = await fetch(parsed.href, {
          headers: { Authorization: `Bearer ${authToken}` }, signal: controller.signal,
        });
        if (!response.ok) throw new Error(`Không tải được media (HTTP ${response.status}).`);
        const blob = await response.blob();
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setObjectUrl(url); setMessage("");
      } catch (error) {
        if (!cancelled && error.name !== "AbortError") setMessage(error.message);
      }
    })();
    return () => { cancelled = true; controller.abort(); if (url) URL.revokeObjectURL(url); setObjectUrl(""); };
  }, [path, authToken, enabled]);

  if (!path) return null;
  if (kind === "video") return (
    <div className="protected-media">
      {!enabled && <button type="button" className="secondary-button" onClick={() => setEnabled(true)}>Phát video (đăng nhập)</button>}
      {enabled && !objectUrl && <span className="small-muted">{message || "Đang tải video an toàn…"}</span>}
      {objectUrl && <video controls preload={preload} src={objectUrl} />}
    </div>
  );
  return <span className="protected-media">{objectUrl ? (openOnClick ? <a href={objectUrl} target="_blank" rel="noreferrer"><img src={objectUrl} alt={alt} className={className} loading="lazy" /></a> : <img src={objectUrl} alt={alt} className={className} loading="lazy" />) : <span className="small-muted">{message || "Đang tải ảnh…"}</span>}</span>;
}
