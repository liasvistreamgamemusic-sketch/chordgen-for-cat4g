// サーバーの API を呼ぶ。誤りは ApiError にそろえる（{"error":{"code","message"}}）。

export class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

/** 403 のときに呼ぶ（画面上部に案内を出す）。 */
let onForbidden = () => {};
let onUnreachable = () => {};
let onReachable = () => {};

export function setHandlers(h) {
  onForbidden = h.forbidden || onForbidden;
  onUnreachable = h.unreachable || onUnreachable;
  onReachable = h.reachable || onReachable;
}

function errorFrom(status, body) {
  const e = body && body.error;
  if (status === 403) onForbidden();
  if (e && typeof e.message === "string") return new ApiError(status, e.code || "ERROR", e.message);
  if (status === 403) return new ApiError(status, "FORBIDDEN", "この画面を開く権限がありません。コンソールに出ている URL から開き直してください。");
  if (status === 404) return new ApiError(status, "NOT_FOUND", "見つかりません。");
  return new ApiError(status, "HTTP_" + status, `サーバーが誤りを返しました（${status}）。`);
}

export async function request(method, path, { json, body } = {}) {
  const init = { method, credentials: "same-origin", headers: { Accept: "application/json" } };
  if (json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(json);
  } else if (body !== undefined) {
    init.body = body;
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (e) {
    onUnreachable();
    throw new ApiError(0, "NETWORK", "サーバーにつながりません。run.bat のコンソールが開いたままか確かめてください。");
  }
  onReachable();
  let data = null;
  const text = await res.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }
  if (!res.ok) throw errorFrom(res.status, data);
  return data;
}

export const api = {
  info: () => request("GET", "/api/info"),
  jobs: () => request("GET", "/api/jobs"),
  job: (id) => request("GET", `/api/jobs/${encodeURIComponent(id)}`),
  cancel: (id) => request("POST", `/api/jobs/${encodeURIComponent(id)}/cancel`),
  outputs: () => request("GET", "/api/outputs"),
  output: (name) => request("GET", `/api/outputs/${encodeURIComponent(name)}`),
  deleteOutput: (name) => request("DELETE", `/api/outputs/${encodeURIComponent(name)}`),
  openFolder: (target, name) => request("POST", "/api/open-folder", { json: name ? { target, name } : { target } }),
  shutdown: () => request("POST", "/api/shutdown"),
  downloadUrl: (name) => `/api/outputs/${encodeURIComponent(name)}?download=1`,
  eventsUrl: (id) => `/api/jobs/${encodeURIComponent(id)}/events`,
};

/**
 * ジョブを送る。送信の進み具合（0〜1）を onProgress に渡す。fetch では upload の進みが取れないので XHR を使う。
 * 戻り値は { id }。
 */
export function submitJob(formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.withCredentials = true;
    xhr.responseType = "text";
    xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    });
    xhr.addEventListener("load", () => {
      onReachable();
      let data = null;
      try {
        data = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        data = null;
      }
      if (xhr.status >= 200 && xhr.status < 300 && data && data.id) resolve(data);
      else reject(errorFrom(xhr.status, data));
    });
    xhr.addEventListener("error", () => {
      onUnreachable();
      reject(new ApiError(0, "NETWORK", "送れませんでした。サーバーが動いているか確かめてください。"));
    });
    xhr.addEventListener("abort", () => reject(new ApiError(0, "ABORTED", "送るのをやめました。")));
    xhr.send(formData);
  });
}
