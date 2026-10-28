import { useEffect, useMemo, useState } from "react";

const API = "http://localhost:3001";

function SecureGradcamImage({ path, token, alt }) {
  const [state, setState] = useState({url:"",error:""});
  useEffect(() => {
    let current = true;
    let currentBlob = "";
    const controller = new AbortController();
    setState({url:"",error:""});
    if (!token || !/^\/api\/gradcam\/real\/(video|camera)\/[1-9][0-9]*\/image\/sample_[0-9]{3}_(original|heat|overlay)\.jpg$/.test(path || "")) {
      setState({url:"",error:"Đường dẫn ảnh không hợp lệ."});
      return () => controller.abort();
    }
    fetch(`${API}${path}`, {headers:{Authorization:`Bearer ${token}`},signal:controller.signal})
      .then(async response => {
        if (!response.ok) throw new Error(`Ảnh không tải được (HTTP ${response.status}).`);
        const blob = await response.blob();
        if (!blob.type.startsWith("image/")) throw new Error("Dữ liệu trả về không phải hình ảnh.");
        return URL.createObjectURL(blob);
      })
      .then(url => {
        if (current) {currentBlob=url;setState({url,error:""});}
        else URL.revokeObjectURL(url);
      })
      .catch(error => {if(current && error.name!=="AbortError") setState({url:"",error:error.message});});
    return () => {
      current=false;controller.abort();if(currentBlob) URL.revokeObjectURL(currentBlob);
    };
  },[path,token]);
  if (state.url) return <img className="real-gradcam-image" src={state.url} alt={alt} loading="lazy" />;
  return <div className="real-gradcam-image-placeholder">{state.error || "Đang tải ảnh..."}</div>;
}

const EYE_LABELS = {
  appear_closed:"Hai mắt có vẻ đang nhắm (một khung hình)",
  appear_open:"Hai mắt có vẻ đang mở (một khung hình)",
  uncertain:"Không đủ căn cứ xác định trạng thái mắt",
  unknown:"Không đo được trạng thái mắt từ khung hình này",
};

const LABELS = {
  angry:"Tức giận",disgust:"Khó chịu",fear:"Sợ hãi",happy:"Vui",
  neutral:"Trung tính",sad:"Buồn",surprise:"Bất ngờ",
};

export default function RealGradcamPage({videos,token}) {
  const [kind,setKind] = useState("video");
  const [videoId,setVideoId] = useState("");
  const [cameraId,setCameraId] = useState("");
  const [cameraList,setCameraList] = useState([]);
  const [cameraError,setCameraError] = useState("");
  const [data,setData] = useState({status:"idle",examples:[]});
  const [message,setMessage] = useState("");
  const [busy,setBusy] = useState(false);
  const selectedId = kind === "video" ? videoId : cameraId;
  const validId = /^[1-9]\d*$/.test(selectedId);
  const videoOptions = useMemo(() => (Array.isArray(videos) ? videos : [])
    .filter(v => v?.class_id && v?.status !== "processing")
    .sort((a,b) => Number(b.id)-Number(a.id)),[videos]);
  useEffect(() => {
    if (!videoId && videoOptions.length) setVideoId(String(videoOptions[0].id));
  },[videoOptions,videoId]);

  useEffect(() => {
    if (!token) return undefined;
    let cancelled=false;
    fetch(`${API}/api/camera-recordings?page=1`,{headers:{Authorization:`Bearer ${token}`}})
      .then(async response => {
        if (!response.ok) throw new Error(`Danh sách camera chưa tải được (HTTP ${response.status}).`);
        return response.json();
      })
      .then(payload=>{if(!cancelled) {
        const items=Array.isArray(payload.items)?payload.items:[];
        setCameraList(items);
        setCameraId(current=> current || (items.length?String(items[0].id):""));
      }})
      .catch(error=>{if(!cancelled) setCameraError(error.message)});
    return ()=>{cancelled=true};
  },[token]);

  useEffect(() => {
    setData({status:"idle",examples:[]});setMessage("");
    if (!validId || !token) return undefined;
    let cancelled=false;
    const refresh = async () => {
      try {
        const response=await fetch(`${API}/api/gradcam/real/${kind}/${selectedId}`,
          {headers:{Authorization:`Bearer ${token}`},cache:"no-store"});
        const body=await response.json().catch(()=>null);
        if (!response.ok) throw new Error(body?.message || `Grad-CAM HTTP ${response.status}`);
        if (!cancelled) {
          setData(body);
          if (body.status === "failed") setMessage(body.message || "Tạo Grad-CAM thất bại.");
          else if (body.status === "ready" || body.status === "empty" || body.status === "idle") setMessage("");
        }
      } catch(error) {if(!cancelled) setMessage(error.message);}
    };
    refresh();
    const timer=setInterval(refresh,4500);
    return ()=>{cancelled=true;clearInterval(timer);};
  },[kind,selectedId,validId,token]);

  const generate = async () => {
    if (!validId || busy || data.status==="processing") return;
    setBusy(true);setMessage("");
    try {
      const response=await fetch(`${API}/api/gradcam/real/${kind}/${selectedId}/generate`,
        {method:"POST",headers:{Authorization:`Bearer ${token}`}});
      const body=await response.json().catch(()=>null);
      if (!response.ok) throw new Error(body?.message || `HTTP ${response.status}`);
      setData({status:"processing",examples:[]});setMessage("");
    } catch(error) {setMessage(error.message);}
    finally {setBusy(false)}
  };

  return <section className="real-gradcam-panel">
    <h2>Grad-CAM từ video / camera thực tế</h2>
    <p>Chọn video đã lưu để tạo Grad-CAM cho dự đoán biểu cảm <strong>FER</strong>. Trạng thái mí mắt (nếu đo được) hiển thị riêng; nhắm mắt không đồng nghĩa "buồn" và mở mắt không đồng nghĩa "trung tính". Không tự xác định danh tính hay trạng thái tâm lý học sinh.</p>
    <div className="real-gradcam-controls">
      <label>Nguồn dữ liệu
        <select value={kind} onChange={e=>setKind(e.target.value)}>
          <option value="video">Video tải lên</option>
          <option value="camera">Camera đã ghi</option>
        </select>
      </label>
      {kind === "video" ? <label>Chọn video
        <select value={videoId} onChange={e=>setVideoId(e.target.value)}>
          {!videoOptions.length && <option value="">Chưa có video</option>}
          {videoOptions.map(v=><option key={v.id} value={String(v.id)}>
            {`#${v.id} — ${v.original_name || "Video"} (${v.status || ""})`}
          </option>)}
        </select>
      </label> : <label>Camera đã lưu (25 phiên gần nhất)
        <select value={cameraId} onChange={e=>setCameraId(e.target.value)}>
          {!cameraList.length && <option value="">Chưa có video camera</option>}
          {cameraList.map(c=><option value={String(c.id)} key={c.id}>
            {`#${c.id} — ${c.class_name || "Lớp"} — ${new Date(c.started_at).toLocaleString("vi-VN")}`}
          </option>)}
        </select>
        {cameraError && <small className="real-gradcam-warning">{cameraError}</small>}
        <small>Video cũ hơn: nhập ID bản ghi từ mục Video camera đã quay.</small>
        <input type="number" min="1" step="1" placeholder="ID camera cũ (nếu có)"
          onChange={e=>{if(e.target.value)setCameraId(e.target.value)}} />
      </label>}
      <button type="button" className="primary-button" disabled={!validId||busy||data.status==="processing"} onClick={generate}>
        {data.status==="processing"?"Đang tạo heatmap...":"Tạo Grad-CAM từ video này"}
      </button>
    </div>
    {message && <p role="status" className="real-gradcam-warning">{message}</p>}
    {data.status==="processing" && <p role="status">Python đang trích khuôn mặt và tính Grad-CAM. Có thể mất vài phút; không ảnh hưởng camera trực tiếp.</p>}
    {data.status==="idle" && <p>Chưa có heatmap cho video này. Nhấn <strong>Tạo Grad-CAM từ video này</strong> để bắt đầu.</p>}
    {data.status==="empty" && <div role="status" className="real-gradcam-summary">
      <strong>Chưa có khuôn mặt đủ điều kiện để tạo Grad-CAM đáng tin cậy.</strong>
      <p>Đã giải mã {data.decoded_frames ?? "?"}/{data.sampled_frames ?? "?"} khung hình; phát hiện {data.detected_face_regions ?? "?"} vùng nghi là mặt. Bỏ qua {data.skipped_quality?.side_or_occluded ?? 0} vùng mặt quá nghiêng/bị che, {data.detector_counts?.rejected_no_eyes ?? 0} vùng không đủ bằng chứng có hai mắt, {data.skipped_quality?.face_too_small ?? 0} vùng quá nhỏ, {data.skipped_quality?.too_blurry ?? 0} vùng mờ, {data.skipped_quality?.too_dark ?? 0} vùng quá tối và {data.skipped_quality?.annotated_video ?? 0} vùng bị nét vẽ/điểm AI đè lên.</p>
      <p>Đây là kết quả lọc mẫu, không chứng minh video không có người. Video <strong>gốc chưa vẽ khung/điểm AI</strong>, khuôn mặt rõ ở góc gần chính diện sẽ phù hợp hơn. Không tự tạo heatmap khi thiếu đầu vào hợp lệ; kiểm tra chỉ số phát hiện và loại mẫu bên trên để tìm nguyên nhân.</p>
    </div>}
    {data.status==="ready" && <>
      <p className="real-gradcam-summary">Đã tạo {data.examples.length} mẫu từ video đang chọn; kiểm tra {data.decoded_frames ?? data.sampled_frames} khung hình. Đã loại {data.skipped_quality?.annotated_video ?? 0} vùng bị đường/điểm AI vẽ đè. Màu đỏ của heatmap <strong>không có nghĩa dự đoán đúng</strong>. Nhãn FER là dự đoán mô hình chưa hiệu chuẩn; trạng thái mắt được đo riêng ở từng khung.</p>
      <div className="real-gradcam-grid">
        {data.examples.map((sample,index)=><article className="real-gradcam-card" key={`${selectedId}-${index}-${sample.time_seconds}`}>
          <strong>Mẫu {index+1} — {Number(sample.time_seconds||0).toFixed(1)} giây</strong>
          <div className="real-gradcam-image-grid">
            <div><span>Ảnh khuôn mặt</span><SecureGradcamImage path={sample.original_url} token={token} alt="Khuôn mặt trích từ video" /></div>
            <div><span>Heatmap</span><SecureGradcamImage path={sample.heatmap_url} token={token} alt="Grad-CAM của mô hình FER" /></div>
            <div><span>Chồng heatmap</span><SecureGradcamImage path={sample.overlay_url} token={token} alt="Overlay Grad-CAM trên ảnh khuôn mặt" /></div>
          </div>
          <p><strong>Quan sát mắt:</strong> {EYE_LABELS[sample.eye_state] || EYE_LABELS.unknown}
          {typeof sample.eye_ear === "number" && <small> (EAR ≈ {sample.eye_ear.toFixed(3)}, chỉ mang tính ước lượng)</small>}</p>
          <p><strong>FER (biểu cảm, không phải trạng thái mắt):</strong> {sample.interpretation === "model_prediction"
            ? <>Mô hình dự đoán <b>{LABELS[sample.pred_label]||sample.pred_label||"Không xác định"}</b> · điểm Softmax {(100*Number(sample.confidence||0)).toFixed(1)}% (chưa hiệu chuẩn).</>
            : <> <b>Không đủ căn cứ kết luận biểu cảm.</b> Nhãn mô hình xếp cao nhất là {LABELS[sample.pred_label]||sample.pred_label||"Không xác định"} với điểm {(100*Number(sample.confidence||0)).toFixed(1)}% — chỉ hiển thị để đối chiếu kỹ thuật.</>}</p>
          <small>Bộ phát hiện mặt: {sample.detector}. Heatmap giải thích dự đoán FER ở khung hình này, không xác nhận nhãn là đúng và không đánh giá mức tập trung hay tâm lý.</small>
        </article>)}
      </div>
    </>}
  </section>;
}
