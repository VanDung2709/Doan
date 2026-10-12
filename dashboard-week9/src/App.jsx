import { Component, useEffect, useMemo, useRef, useState } from "react";
import { io } from "socket.io-client";
import "./index.css";
import ProtectedMedia from "./ProtectedMedia";
import CameraRecordings from "./CameraRecordings";

const API = "http://localhost:3001";
const HISTORY_PAGE_SIZE = 200;

function mergeHistoryRows(currentRows, incomingRows) {
  const byId = new Map();
  // Newer server rows win: a teacher may have just confirmed a Track mapping.
  [...currentRows, ...incomingRows].forEach((row) => {
    if (row?.id !== null && row?.id !== undefined) {
      byId.set(String(row.id), row);
    }
  });
  return [...byId.values()].sort((a, b) => Number(b.id) - Number(a.id));
}

const toDateInput = (date) => {
  const local = new Date(
    date.getTime() -
      date.getTimezoneOffset() * 60000
  );

  return local.toISOString().slice(0, 10);
};

const makeDefaultTrendFrom = () => {
  const date = new Date();
  date.setDate(date.getDate() - 29);
  return toDateInput(date);
};

// Recover from a page-level render error without blanking the whole Dashboard.
// Switching to another menu page remounts this boundary (key={activePage}).
class PageErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error, info) {
    console.error("[Dashboard page error]", error, info);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <section className="dashboard-section" role="alert">
        <h2>Không thể hiển thị trang này</h2>
        <p>Hãy chọn trang khác trên menu hoặc tải lại trang. Lỗi đã được ghi trong Console (F12).</p>
        <button className="secondary-button" type="button" onClick={() => window.location.reload()}>
          Tải lại Dashboard
        </button>
      </section>
    );
  }
}

function App() {
  const [token, setToken] = useState(
    localStorage.getItem("token")
  );

  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem("user");
      return saved ? JSON.parse(saved) : null;
    } catch (error) {
      console.warn("Thông tin người dùng trong trình duyệt không hợp lệ.", error);
      return null;
    }
  });

  const [activePage, setActivePage] = useState("overview");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loginMessage, setLoginMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const [stats, setStats] = useState({
    total_classes: 0,
    total_students: 0,
    alerts_24h: 0,
    total_videos: 0,
    active_tracks: 0,
    distracted: 0,
    anxiety: 0,
  });

  const [allAlerts, setAllAlerts] = useState([]);
  const [aiResults, setAiResults] = useState([]);
  const [classes, setClasses] = useState([]);
  const [students, setStudents] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [selectedClassId, setSelectedClassId] = useState("");
  const [cameraClassId, setCameraClassId] = useState("");
  const [cameraSessionId, setCameraSessionId] = useState("");
  const [alertClassId, setAlertClassId] = useState("");
  const [alertSessionId, setAlertSessionId] = useState("");
  const [historyClassId, setHistoryClassId] = useState("");
  const [historySessionId, setHistorySessionId] = useState("");
  const [historyResults, setHistoryResults] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyLoadingMore, setHistoryLoadingMore] = useState(false);
  const [historyHasMore, setHistoryHasMore] = useState(false);
  const [historyLoadError, setHistoryLoadError] = useState("");
  const historyReachedEndRef = useRef(false);
  const [statsClassId, setStatsClassId] = useState("");
  const [statsSessionId, setStatsSessionId] = useState("");
  const [statsStudentId, setStatsStudentId] = useState("");
  const [studentReport, setStudentReport] = useState(null);
  const [videoClassId, setVideoClassId] = useState("");
  const [videoSessionId, setVideoSessionId] = useState("");
  const [gradcamExamples, setGradcamExamples] = useState([]);
  const [videos, setVideos] = useState([]);
  const [users, setUsers] = useState([]);
  const [trendFrom, setTrendFrom] = useState(
    makeDefaultTrendFrom
  );
  const [trendTo, setTrendTo] = useState(
    () => toDateInput(new Date())
  );
  const [trendPreset, setTrendPreset] = useState("30");
  const [trends, setTrends] = useState([]);

  const [health, setHealth] = useState({
    backend: false,
    database: false,
  });

  const [toast, setToast] = useState("");
  const [errorText, setErrorText] = useState("");

  const [alertFilter, setAlertFilter] = useState("all");
  const [historySearch, setHistorySearch] = useState("");

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const fileInputRef = useRef(null);
  const socketRef = useRef(null);
  const authorizedClassesRef = useRef([]);

  const isAdmin = user?.role === "admin";
  // Theo đề cương: giáo viên được quản lý lớp học và danh sách học sinh;
  // quản lý tài khoản/phân quyền toàn hệ thống vẫn chỉ dành cho Admin.
  const canManageClassroom = ["admin", "teacher"].includes(user?.role);
  const historyVideoMatch = historySearch.trim().match(/^video\s+(\d+)$/i);
  const historyVideoId = historyVideoMatch ? historyVideoMatch[1] : "";

  const authHeaders = useMemo(
    () => ({
      Authorization: `Bearer ${token}`,
    }),
    [token]
  );

  const jsonHeaders = useMemo(
    () => ({
      ...authHeaders,
      "Content-Type": "application/json",
    }),
    [authHeaders]
  );

  const recentAlerts = useMemo(
    () => allAlerts.slice(0, 5),
    [allAlerts]
  );

  const filteredAlerts = useMemo(() => {
    return allAlerts.filter((item) => {
      const typeOk =
        alertFilter === "all"
        || item.alert_type === alertFilter;

      const classOk =
        !alertClassId
        || String(item.class_id) === String(alertClassId);

      const sessionOk =
        !alertSessionId
        || String(item.session_id) === String(alertSessionId);

      return typeOk && classOk && sessionOk;
    });
  }, [
    allAlerts,
    alertFilter,
    alertClassId,
    alertSessionId,
  ]);

  const latestTracks = useMemo(() => {
    if (!cameraClassId || !cameraSessionId) {
      return [];
    }

    const map = new Map();
    const now = Date.now();

    aiResults
      .filter(
        (item) =>
          !item.video_id
          && String(item.class_id) === String(cameraClassId)
          && String(item.session_id) === String(cameraSessionId)
      )
      .forEach((item) => {
        if (item.track_id === null || item.track_id === undefined) {
          return;
        }

        if (!map.has(item.track_id)) {
          map.set(item.track_id, item);
        }
      });

    return [...map.values()]
      .filter((item) => {
        const time = new Date(
          item.captured_at || item.created_at
        ).getTime();

        return Number.isFinite(time)
          && now - time <= 5 * 60 * 1000;
      })
      .sort((a, b) => Number(a.track_id) - Number(b.track_id));
  }, [aiResults, cameraClassId, cameraSessionId]);

  const cameraOnline = useMemo(() => {
    if (!cameraClassId || !cameraSessionId) {
      return false;
    }

    const latest = aiResults.find(
      (item) =>
        !item.video_id
        && String(item.class_id) === String(cameraClassId)
        && String(item.session_id) === String(cameraSessionId)
    );

    if (!latest) {
      return false;
    }

    const t = new Date(
      latest.captured_at || latest.created_at
    ).getTime();

    return Number.isFinite(t)
      && Date.now() - t <= 15000;
  }, [aiResults, cameraClassId, cameraSessionId]);

  const filteredHistory = useMemo(() => {
    const q = historySearch.trim().toLowerCase();

    return historyResults.filter((item) => {
      if (
        historyClassId
        && String(item.class_id) !== String(historyClassId)
      ) {
        return false;
      }

      if (
        historySessionId
        && String(item.session_id) !== String(historySessionId)
      ) {
        return false;
      }

      if (!q) {
        return true;
      }

      const text = [
        item.track_id,
        item.student_name,
        item.fusion_status,
        item.fusion_reason,
        item.emotion,
        item.engagement,
        item.class_name,
        item.session_title,
        item.video_name,
        item.video_id
          ? `video ${item.video_id}`
          : "camera",
      ]
        .join(" ")
        .toLowerCase();

      return text.includes(q);
    });
  }, [
    historyResults,
    historySearch,
    historyClassId,
    historySessionId,
  ]);

  const formatAlertType = (type) => {
    if (type === "mat_tap_trung") {
      return "Mất tập trung";
    }

    if (type === "dau_hieu_lo_au") {
      return "Dấu hiệu cần quan sát thêm";
    }

    return type || "Cảnh báo";
  };

  const formatFusionStatus = (status) => {
    const value = String(status || "")
      .replaceAll("_", " ")
      .toUpperCase();

    if (value === "DAU HIEU MAT TAP TRUNG") {
      return "Mất tập trung";
    }

    if (value === "DAU HIEU LO AU") {
      return "Dấu hiệu cần quan sát thêm";
    }

    if (value === "BINH THUONG") {
      return "Bình thường";
    }

    if (value === "THEO DOI") {
      return "Theo dõi";
    }

    return status || "Chưa có";
  };

  const fusionClass = (status) => {
    const value = String(status || "")
      .replaceAll("_", " ")
      .toUpperCase();

    if (value.includes("LO AU")) {
      return "badge anxiety";
    }

    if (value.includes("MAT TAP TRUNG")) {
      return "badge distracted";
    }

    if (value === "BINH THUONG") {
      return "badge normal";
    }

    return "badge watch";
  };

  const videoStatusText = (status) => {
    if (status === "uploaded") {
      return "Đã tải";
    }

    if (status === "processing") {
      return "Đang phân tích";
    }

    if (status === "completed") {
      return "Hoàn thành";
    }

    if (status === "failed") {
      return "Thất bại";
    }

    return status || "Chưa rõ";
  };

  const apiJson = async (
    path,
    options = {}
  ) => {
    const response = await fetch(
      `${API}${path}`,
      options
    );

    let data = null;

    try {
      data = await response.json();
    } catch {
      data = null;
    }

    if (response.status === 401) {
      handleLogout();
      throw new Error(
        "Phiên đăng nhập đã hết hạn."
      );
    }

    if (!response.ok) {
      throw new Error(
        data?.message ||
          `Lỗi HTTP ${response.status}`
      );
    }

    return data;
  };

  const showToast = (message) => {
    setToast(message);
    window.setTimeout(
      () => setToast(""),
      2800
    );
  };

  const loadHealth = async () => {
    try {
      const response = await fetch(
        `${API}/api/health`
      );

      const data = await response.json();

      setHealth({
        backend: Boolean(
          response.ok && data.ok
        ),
        database: Boolean(
          response.ok &&
            data.ok &&
            data.database === "PostgreSQL"
        ),
      });
    } catch {
      setHealth({
        backend: false,
        database: false,
      });
    }
  };

  const loadStats = async () => {
    const data = await apiJson(
      "/api/stats/summary",
      {
        headers: authHeaders,
      }
    );

    setStats(data);
  };

  const loadAlerts = async () => {
    const data = await apiJson(
      "/api/alerts?limit=1000",
      {
        headers: authHeaders,
      }
    );

    setAllAlerts(data);
  };

  const loadAiResults = async () => {
    const data = await apiJson(
      "/api/ai/results?limit=500",
      {
        headers: authHeaders,
      }
    );

    setAiResults(data);
  };

  const loadClasses = async () => {
    const data = await apiJson(
      "/api/classes",
      {
        headers: authHeaders,
      }
    );

    setClasses(data);

    if (data.length === 0) {
      setSelectedClassId("");
      return;
    }

    const selectedStillExists =
      data.some(
        (item) =>
          String(item.id) ===
          String(selectedClassId)
      );

    if (!selectedStillExists) {
      setSelectedClassId(
        String(data[0].id)
      );
    }
  };

  const loadSessions = async () => {
    const data = await apiJson(
      "/api/sessions",
      {
        headers: authHeaders,
      }
    );

    setSessions(data);
  };

  const loadCameraContext = async () => {
    const data = await apiJson(
      "/api/runtime/camera-context",
      { headers: authHeaders }
    );

    setCameraClassId(
      data?.class?.id
        ? String(data.class.id)
        : ""
    );

    setCameraSessionId(
      data?.session?.id
        ? String(data.session.id)
        : ""
    );
  };

  const loadStudents = async () => {
    const data = await apiJson(
      "/api/students",
      {
        headers: authHeaders,
      }
    );

    setStudents(data);
  };

  const loadGradcam = async () => {
    const data = await apiJson(
      "/api/gradcam/examples",
      {
        headers: authHeaders,
      }
    );

    setGradcamExamples(data);
  };

  const loadVideos = async () => {
    const data = await apiJson(
      "/api/videos",
      {
        headers: authHeaders,
      }
    );

    setVideos(data);
  };

  const loadUsers = async () => {
    if (!isAdmin) {
      setUsers([]);
      return;
    }

    const data = await apiJson(
      "/api/users",
      {
        headers: authHeaders,
      }
    );

    setUsers(data);
  };

  const loadTrends = async (
    from = trendFrom,
    to = trendTo
  ) => {
    const params = new URLSearchParams({
      from,
      to,
    });

    if (statsClassId) {
      params.set("classId", statsClassId);
    }

    if (statsSessionId) {
      params.set("sessionId", statsSessionId);
    }

    if (statsStudentId) {
      params.set("studentId", statsStudentId);
      params.set("videoOnly", "1");
    }

    const data = await apiJson(
      `/api/stats/trends?${params.toString()}`,
      {
        headers: authHeaders,
      }
    );

    setTrends(data);
  };

  const loadStudentReport = async (
    studentId = statsStudentId
  ) => {
    if (!studentId) {
      setStudentReport(null);
      return;
    }

    const params = new URLSearchParams({
      studentId,
      from: trendFrom,
      to: trendTo,
    });

    if (statsClassId) {
      params.set("classId", statsClassId);
    }

    if (statsSessionId) {
      params.set("sessionId", statsSessionId);
    }

    const data = await apiJson(
      `/api/stats/student-detail?${params.toString()}`,
      {
        headers: authHeaders,
      }
    );

    setStudentReport(data);
  };

  const applyTrendPreset = (preset) => {
    const today = new Date();
    let from = new Date(today);
    let to = new Date(today);

    if (preset === "7") {
      from.setDate(from.getDate() - 6);
    }

    if (preset === "30") {
      from.setDate(from.getDate() - 29);
    }

    if (preset === "month") {
      from = new Date(
        today.getFullYear(),
        today.getMonth(),
        1
      );
    }

    if (preset === "prev-month") {
      from = new Date(
        today.getFullYear(),
        today.getMonth() - 1,
        1
      );

      to = new Date(
        today.getFullYear(),
        today.getMonth(),
        0
      );
    }

    if (preset === "all") {
      const alertDates = allAlerts
        .filter(
          (item) =>
            (
              !statsClassId
              || String(item.class_id) === String(statsClassId)
            )
            && (
              !statsSessionId
              || String(item.session_id) === String(statsSessionId)
            )
            && (
              !statsStudentId
              || String(item.student_id) === String(statsStudentId)
            )
            && (
              !statsStudentId
              || item.video_id
            )
        )
        .map((item) => new Date(item.created_at))
        .filter(
          (date) =>
            !Number.isNaN(date.getTime())
        );

      if (alertDates.length > 0) {
        from = new Date(
          Math.min(
            ...alertDates.map(
              (date) => date.getTime()
            )
          )
        );
      }
    }

    setTrendPreset(preset);
    setTrendFrom(toDateInput(from));
    setTrendTo(toDateInput(to));
  };

  const refreshAll = async () => {
    setErrorText("");

    try {
      await Promise.all([
        loadHealth(),
        loadStats(),
        loadAlerts(),
        loadAiResults(),
        loadClasses(),
        loadSessions(),
        loadCameraContext(),
        loadStudents(),
        loadGradcam(),
        loadVideos(),
        loadUsers(),
        loadTrends(),
      ]);
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();

    setLoginMessage("");
    setLoading(true);

    try {
      const response = await fetch(
        `${API}/api/auth/login`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify({
            email,
            password,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.message ||
            "Đăng nhập thất bại"
        );
      }

      localStorage.setItem(
        "token",
        data.token
      );

      localStorage.setItem(
        "user",
        JSON.stringify(data.user)
      );

      setToken(data.token);
      setUser(data.user);
      setActivePage("overview");
    } catch (error) {
      setLoginMessage(error.message);
    } finally {
      setLoading(false);
    }
  };

  function handleLogout() {
    localStorage.removeItem("token");
    localStorage.removeItem("user");

    setToken(null);
    setUser(null);
    setEmail("");
    setPassword("");
    setAllAlerts([]);
    setAiResults([]);
    setClasses([]);
    setStudents([]);
    setSessions([]);
    setCameraClassId("");
    setCameraSessionId("");
    setAlertClassId("");
    setAlertSessionId("");
    setHistoryClassId("");
    setHistorySessionId("");
    setHistoryResults([]);
    setHistoryLoading(false);
    setHistoryLoadingMore(false);
    setHistoryHasMore(false);
    setHistoryLoadError("");
    setStatsClassId("");
    setStatsSessionId("");
    setStatsStudentId("");
    setStudentReport(null);
    setVideoClassId("");
    setVideoSessionId("");
    setVideos([]);
    setUsers([]);
    setGradcamExamples([]);
  }

  useEffect(() => {
    if (!token) {
      return undefined;
    }

    refreshAll();

    const refreshTimer =
      window.setInterval(
        () => {
          Promise.all([
            loadHealth(),
            loadStats(),
            loadAlerts(),
            loadAiResults(),
          ]).catch((error) => {
            setErrorText(error.message);
          });
        },
        5000
      );

    return () => {
      window.clearInterval(
        refreshTimer
      );
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  useEffect(() => {
    if (!token) {
      return undefined;
    }

    const socket = io(API, {
      auth: {
        token,
      },
      reconnection: true,
      reconnectionAttempts: 20,
      reconnectionDelay: 1000,
    });

    socketRef.current = socket;
    socket.on("connect", () => {
      authorizedClassesRef.current.forEach((item) => socket.emit("join_class", { classId: item.id }));
    });
    socket.on("ai_result", (data) => {
      setAiResults((prev) => [
        data,
        ...prev.filter(
          (item) => item.id !== data.id
        ),
      ].slice(0, 500));

      loadStats().catch(() => {});
    });

    socket.on(
      "alert_created",
      (data) => {
        setAllAlerts((prev) => [
          data,
          ...prev.filter(
            (item) =>
              item.id !== data.id
          ),
        ]);

        loadStats().catch(() => {});
      }
    );

    socket.on(
      "video_status",
      () => {
        loadVideos().catch(() => {});
      }
    );

    socket.on(
      "video_progress",
      (data) => {
        setVideos((prev) =>
          prev.map((item) =>
            Number(item.id) === Number(data.id)
              ? {
                  ...item,
                  status: "processing",
                  progress:
                    Number(data.progress) || 0,
                }
              : item
          )
        );
      }
    );

    socket.on(
      "connect_error",
      (error) => {
        console.warn(
          "Socket:",
          error.message
        );
      }
    );

    return () => {
      socket.close();
      if (socketRef.current === socket) socketRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  // Join authorized rooms when the list of classes changes. The backend checks membership.
  useEffect(() => {
    authorizedClassesRef.current = classes;
    if (!token || !socketRef.current?.connected) return;
    classes.forEach((item) => socketRef.current.emit("join_class", { classId: item.id }));
  }, [token, classes]);

  // Lich su AI doc tu PostgreSQL theo tung trang, doc lap cache realtime 500 ban ghi.
  // Du lieu live co the het han tren bang Realtime, nhung cac ban ghi lich su van duoc giu trong DB.
  useEffect(() => {
    if (!token || activePage !== "history") {
      return undefined;
    }

    let cancelled = false;
    let inFlight = false;
    setHistoryResults([]);
    setHistoryHasMore(false);
    historyReachedEndRef.current = false;
    setHistoryLoadError("");

    const makeHistoryParams = (beforeId = "") => {
      const params = new URLSearchParams({
        limit: String(HISTORY_PAGE_SIZE),
      });
      if (historyClassId) params.set("classId", historyClassId);
      if (historySessionId) params.set("sessionId", historySessionId);
      if (historyVideoId) params.set("videoId", historyVideoId);
      if (beforeId) params.set("beforeId", beforeId);
      return params;
    };

    const loadLatestHistory = async (showLoading = false) => {
      if (inFlight) return;
      inFlight = true;
      if (showLoading && !cancelled) setHistoryLoading(true);

      try {
        const params = makeHistoryParams();
        const data = await apiJson(
          `/api/ai/results?${params.toString()}`,
          { headers: authHeaders }
        );
        if (!Array.isArray(data)) {
          throw new Error("API lịch sử AI trả về dữ liệu không hợp lệ.");
        }
        if (!cancelled) {
          if (showLoading) {
            setHistoryResults(data);
          } else {
            setHistoryResults((prev) => mergeHistoryRows(prev, data));
          }
          setHistoryHasMore(
            data.length === HISTORY_PAGE_SIZE && !historyReachedEndRef.current
          );
          setHistoryLoadError("");
        }
      } catch (error) {
        if (!cancelled) {
          setHistoryLoadError(error.message || "Không tải được lịch sử AI.");
        }
      } finally {
        inFlight = false;
        if (showLoading && !cancelled) setHistoryLoading(false);
      }
    };

    loadLatestHistory(true);
    const timer = window.setInterval(() => loadLatestHistory(false), 10000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [
    token,
    activePage,
    historyClassId,
    historySessionId,
    historyVideoId,
    authHeaders,
    // apiJson is recreated inside App; intentionally omitted to avoid refetch loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  ]);

  const loadOlderHistory = async () => {
    if (!token || historyLoadingMore || !historyHasMore || historyResults.length === 0) {
      return;
    }

    const oldestId = historyResults.reduce((minId, row) => {
      const id = Number(row.id);
      return Number.isFinite(id) && (minId === null || id < minId) ? id : minId;
    }, null);

    if (oldestId === null) {
      setHistoryHasMore(false);
      return;
    }

    setHistoryLoadingMore(true);
    setHistoryLoadError("");
    try {
      const params = new URLSearchParams({
        limit: String(HISTORY_PAGE_SIZE),
        beforeId: String(oldestId),
      });
      if (historyClassId) params.set("classId", historyClassId);
      if (historySessionId) params.set("sessionId", historySessionId);
      if (historyVideoId) params.set("videoId", historyVideoId);

      const data = await apiJson(
        `/api/ai/results?${params.toString()}`,
        { headers: authHeaders }
      );
      if (!Array.isArray(data)) {
        throw new Error("API lịch sử AI trả về dữ liệu không hợp lệ.");
      }
      setHistoryResults((prev) => mergeHistoryRows(prev, data));
      if (data.length < HISTORY_PAGE_SIZE) {
        historyReachedEndRef.current = true;
      }
      setHistoryHasMore(data.length === HISTORY_PAGE_SIZE);
    } catch (error) {
      setHistoryLoadError(error.message || "Không tải được bản ghi cũ hơn.");
    } finally {
      setHistoryLoadingMore(false);
    }
  };

  useEffect(() => {
    if (!token) {
      return;
    }

    Promise.all([
      loadTrends(
        trendFrom,
        trendTo
      ),
      loadStudentReport(),
    ]).catch(
      (error) =>
        setErrorText(error.message)
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    trendFrom,
    trendTo,
    statsClassId,
    statsSessionId,
    statsStudentId,
    token,
  ]);

  const changeCameraClass = async (classId) => {
    if (!classId) {
      setCameraClassId("");
      setCameraSessionId("");
      return;
    }

    try {
      const data = await apiJson(
        "/api/runtime/camera-class",
        {
          method: "PUT",
          headers: jsonHeaders,
          body: JSON.stringify({
            classId: Number(classId),
          }),
        }
      );

      setCameraClassId(String(data.class.id));
      setCameraSessionId("");
      showToast(
        "Đã chọn lớp. Hãy chọn tiếp buổi học."
      );
      await loadAiResults();
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const changeCameraSession = async (
    sessionId
  ) => {
    if (!cameraClassId) {
      setErrorText(
        "Hãy chọn lớp trước."
      );
      return;
    }

    if (!sessionId) {
      setCameraSessionId("");
      return;
    }

    try {
      const data = await apiJson(
        "/api/runtime/camera-context",
        {
          method: "PUT",
          headers: jsonHeaders,
          body: JSON.stringify({
            classId:
              Number(cameraClassId),
            sessionId:
              Number(sessionId),
          }),
        }
      );

      setCameraClassId(
        String(data.class.id)
      );
      setCameraSessionId(
        String(data.session.id)
      );

      showToast(data.message);
      await loadAiResults();
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const createSession = async (
    classId,
    onCreated
  ) => {
    if (!classId) {
      setErrorText(
        "Hãy chọn lớp trước khi tạo buổi học."
      );
      return;
    }

    const title = window.prompt(
      "Tên buổi học (ví dụ: Buổi 1 - Chương 3):"
    );

    if (!title) {
      return;
    }

    const today =
      toDateInput(new Date());

    const sessionDate =
      window.prompt(
        "Ngày học (YYYY-MM-DD):",
        today
      );

    if (!sessionDate) {
      return;
    }

    const startTime =
      window.prompt(
        "Giờ bắt đầu (HH:MM, có thể để trống):",
        ""
      ) || "";

    const endTime =
      window.prompt(
        "Giờ kết thúc (HH:MM, có thể để trống):",
        ""
      ) || "";

    try {
      const data = await apiJson(
        `/api/classes/${classId}/sessions`,
        {
          method: "POST",
          headers: jsonHeaders,
          body: JSON.stringify({
            title,
            sessionDate,
            startTime,
            endTime,
          }),
        }
      );

      await loadSessions();

      if (onCreated) {
        onCreated(String(data.id));
      }

      showToast("Đã tạo buổi học.");
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const createManagedSession = async (values) => {
    try {
      await apiJson(
        `/api/classes/${Number(values.classId)}/sessions`,
        {
          method: "POST",
          headers: jsonHeaders,
          body: JSON.stringify({
            title: values.title.trim(),
            sessionDate: values.sessionDate,
            startTime: values.startTime || "",
            endTime: values.endTime || "",
            note: values.note || "",
          }),
        }
      );
      await loadSessions();
      showToast("Đã thêm buổi học.");
      return true;
    } catch (error) {
      setErrorText(error.message);
      return false;
    }
  };

  const updateManagedSession = async (sessionId, values) => {
    try {
      await apiJson(
        `/api/sessions/${Number(sessionId)}`,
        {
          method: "PUT",
          headers: jsonHeaders,
          body: JSON.stringify({
            title: values.title.trim(),
            sessionDate: values.sessionDate,
            startTime: values.startTime || "",
            endTime: values.endTime || "",
            note: values.note || "",
          }),
        }
      );
      await loadSessions();
      showToast("Đã cập nhật buổi học.");
      return true;
    } catch (error) {
      setErrorText(error.message);
      return false;
    }
  };

  const deleteManagedSession = async (session) => {
    try {
      await apiJson(
        `/api/sessions/${Number(session.id)}`,
        {
          method: "DELETE",
          headers: authHeaders,
        }
      );
      await loadSessions();
      const deletedId = String(session.id);
      if (String(cameraSessionId) === deletedId) setCameraSessionId("");
      if (String(videoSessionId) === deletedId) setVideoSessionId("");
      if (String(historySessionId) === deletedId) setHistorySessionId("");
      if (String(alertSessionId) === deletedId) setAlertSessionId("");
      if (String(statsSessionId) === deletedId) setStatsSessionId("");
      showToast("Đã xóa buổi học. Lịch sử AI vẫn còn nhưng không còn gắn với buổi này.");
      return true;
    } catch (error) {
      setErrorText(error.message);
      return false;
    }
  };

  const assignStudentToTrack = async (
    row
  ) => {
    if (
      !row.session_id
      || row.track_id === null
      || row.track_id === undefined
    ) {
      setErrorText(
        "Dữ liệu này chưa có Buổi học hoặc Track ID."
      );
      return;
    }

    const candidates =
      students.filter(
        (item) =>
          String(item.class_id)
          === String(row.class_id)
      );

    if (candidates.length === 0) {
      setErrorText(
        "Lớp này chưa có học sinh để gán."
      );
      return;
    }

    const choices =
      candidates
        .map(
          (item) =>
            `${item.id}: ${item.full_name}${
              item.student_code
                ? ` (${item.student_code})`
                : ""
            }`
        )
        .join("\n");

    const input = window.prompt(
      `Gán Track ${row.track_id} cho học sinh:\n\n${choices}\n\nNhập ID học sinh:`
    );

    if (!input) {
      return;
    }

    if (!row.analysis_run_id) {
      setErrorText("Bản ghi này chưa có mã lượt phân tích. Hãy dùng dữ liệu mới và trang Sơ đồ chỗ ngồi để xác nhận.");
      return;
    }

    const student =
      candidates.find(
        (item) =>
          String(item.id)
          === String(input).trim()
      );

    if (!student) {
      setErrorText(
        "ID học sinh không hợp lệ."
      );
      return;
    }

    try {
      const data = await apiJson(
        `/api/sessions/${row.session_id}/tracks/${row.track_id}/student`,
        {
          method: "PUT",
          headers: jsonHeaders,
          body: JSON.stringify({
            studentId:
              Number(student.id),
            analysisRunId: row.analysis_run_id,
            videoId:
              row.video_id
                ? Number(row.video_id)
                : null,
          }),
        }
      );

      await Promise.all([
        loadAiResults(),
        loadAlerts(),
      ]);
      if (activePage === "history") {
        setHistoryResults((prev) => prev.map((item) =>
          item.session_id === row.session_id
          && item.track_id === row.track_id
          && item.analysis_run_id === row.analysis_run_id
          && item.video_id === row.video_id
          ? { ...item, student_id: student.id, student_name: student.full_name, student_code: student.student_code }
          : item
        ));
      }
      showToast(data.message);
    } catch (error) {
      setErrorText(error.message);
    }
  };


  const addClass = async () => {
    const name = window.prompt(
      "Tên lớp học:"
    );

    if (!name) {
      return;
    }

    const description =
      window.prompt(
        "Mô tả lớp (có thể để trống):"
      ) || "";

    try {
      await apiJson(
        "/api/classes",
        {
          method: "POST",
          headers: jsonHeaders,
          body: JSON.stringify({
            name,
            description,
          }),
        }
      );

      await Promise.all([
        loadClasses(),
        loadStats(),
      ]);

      showToast("Đã thêm lớp.");
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const editClass = async (
    item
  ) => {
    const name = window.prompt(
      "Tên lớp:",
      item.name
    );

    if (!name) {
      return;
    }

    const description =
      window.prompt(
        "Mô tả:",
        item.description || ""
      );

    if (description === null) {
      return;
    }

    try {
      await apiJson(
        `/api/classes/${item.id}`,
        {
          method: "PUT",
          headers: jsonHeaders,
          body: JSON.stringify({
            name,
            description,
          }),
        }
      );

      await loadClasses();
      showToast("Đã cập nhật lớp.");
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const deleteClass = async (
    item
  ) => {
    const ok = window.confirm(
      `Xóa lớp "${item.name}"? Học sinh và các buổi học thuộc lớp này cũng sẽ bị xóa. Bản ghi lịch sử/cảnh báo vẫn có thể còn nhưng mất liên kết với lớp hoặc buổi. Bạn có chắc không?`
    );

    if (!ok) {
      return;
    }

    try {
      await apiJson(
        `/api/classes/${item.id}`,
        {
          method: "DELETE",
          headers: authHeaders,
        }
      );

      await Promise.all([
        loadClasses(),
        loadStudents(),
        loadSessions(),
        loadStats(),
      ]);

      showToast("Đã xóa lớp.");
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const addStudent = async () => {
    if (!selectedClassId) {
      setErrorText(
        "Hãy chọn lớp trước."
      );
      return;
    }

    const selectedClass =
      classes.find(
        (item) =>
          String(item.id) ===
          String(selectedClassId)
      );

    if (!selectedClass) {
      setErrorText(
        "Lớp đang chọn không còn tồn tại. Danh sách lớp sẽ được tải lại."
      );

      await loadClasses();
      return;
    }

    const studentCode =
      window.prompt(
        "Mã học sinh:"
      ) || "";

    const fullName =
      window.prompt(
        "Họ tên học sinh:"
      );

    if (!fullName) {
      return;
    }

    try {
      await apiJson(
        `/api/classes/${selectedClassId}/students`,
        {
          method: "POST",
          headers: jsonHeaders,
          body: JSON.stringify({
            studentCode,
            fullName,
          }),
        }
      );

      await Promise.all([
        loadStudents(),
        loadClasses(),
        loadStats(),
      ]);

      showToast(
        "Đã thêm học sinh."
      );
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const editStudent = async (
    item
  ) => {
    if (classes.length === 0) {
      setErrorText(
        "Chưa có lớp học để gán cho học sinh."
      );
      return;
    }

    const classChoices =
      classes
        .map(
          (classItem) =>
            `${classItem.id}: ${classItem.name}`
        )
        .join("\n");

    const currentClass =
      classes.find(
        (classItem) =>
          Number(classItem.id) ===
          Number(item.class_id)
      );

    const classInput =
      window.prompt(
        `Chọn lớp bằng ID hoặc nhập đúng tên lớp:\n\n${classChoices}`,
        currentClass
          ? String(currentClass.id)
          : String(selectedClassId || "")
      );

    if (classInput === null) {
      return;
    }

    const normalizedInput =
      classInput.trim();

    const selectedClass =
      classes.find(
        (classItem) =>
          String(classItem.id) ===
            normalizedInput
          ||
          String(classItem.name)
            .trim()
            .toLowerCase() ===
            normalizedInput.toLowerCase()
      );

    if (!selectedClass) {
      setErrorText(
        "Lớp bạn chọn không tồn tại. Hãy chọn đúng lớp trong danh sách."
      );
      return;
    }

    const studentCode =
      window.prompt(
        "Mã học sinh:",
        item.student_code || ""
      );

    if (studentCode === null) {
      return;
    }

    const fullName =
      window.prompt(
        "Họ tên:",
        item.full_name
      );

    if (!fullName) {
      return;
    }

    try {
      await apiJson(
        `/api/students/${item.id}`,
        {
          method: "PUT",
          headers: jsonHeaders,
          body: JSON.stringify({
            classId:
              Number(selectedClass.id),
            studentCode,
            fullName,
          }),
        }
      );

      await Promise.all([
        loadStudents(),
        loadClasses(),
        loadStats(),
      ]);

      setSelectedClassId(
        String(selectedClass.id)
      );

      showToast(
        "Đã cập nhật học sinh."
      );
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const deleteStudent = async (
    item
  ) => {
    const ok = window.confirm(
      `Xóa học sinh "${item.full_name}"? Liên kết học sinh với Track sẽ bị xóa; lịch sử AI/cảnh báo có thể không còn hiện tên học sinh. Bạn có chắc không?`
    );

    if (!ok) {
      return;
    }

    try {
      await apiJson(
        `/api/students/${item.id}`,
        {
          method: "DELETE",
          headers: authHeaders,
        }
      );

      await Promise.all([
        loadStudents(),
        loadClasses(),
        loadStats(),
      ]);

      showToast(
        "Đã xóa học sinh."
      );
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const uploadVideo = async (
    event
  ) => {
    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    if (!videoClassId) {
      setErrorText(
        "Hãy chọn lớp học trước khi tải video."
      );
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    if (!videoSessionId) {
      setErrorText(
        "Hãy chọn buổi học trước khi tải video."
      );
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    const formData =
      new FormData();

    formData.append(
      "video",
      file
    );
    formData.append(
      "classId",
      videoClassId
    );
    formData.append(
      "sessionId",
      videoSessionId
    );

    try {
      const response = await fetch(
        `${API}/api/videos/upload`,
        {
          method: "POST",
          headers: authHeaders,
          body: formData,
        }
      );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.message ||
            "Upload thất bại."
        );
      }

      await Promise.all([
        loadVideos(),
        loadStats(),
      ]);

      showToast(
        "Đã tải video."
      );
    } catch (error) {
      setErrorText(error.message);
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const analyzeVideo = async (
    item
  ) => {
    const classId =
      item.class_id || videoClassId;
    const sessionId =
      item.session_id || videoSessionId;

    if (!classId) {
      setErrorText(
        "Hãy chọn lớp học trước khi phân tích video."
      );
      return;
    }

    if (!sessionId) {
      setErrorText(
        "Hãy chọn buổi học trước khi phân tích video."
      );
      return;
    }

    try {
      const data = await apiJson(
        `/api/videos/${item.id}/analyze`,
        {
          method: "POST",
          headers: jsonHeaders,
          body: JSON.stringify({
            classId: Number(classId),
            sessionId: Number(sessionId),
          }),
        }
      );

      showToast(data.message);
      await loadVideos();
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const deleteVideo = async (
    item
  ) => {
    const ok = window.confirm(
      `Xóa video "${item.original_name}"?`
    );

    if (!ok) {
      return;
    }

    try {
      await apiJson(
        `/api/videos/${item.id}`,
        {
          method: "DELETE",
          headers: authHeaders,
        }
      );

      await Promise.all([
        loadVideos(),
        loadStats(),
      ]);

      showToast("Đã xóa video.");
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const changePassword = async (
    event
  ) => {
    event.preventDefault();

    if (
      newPassword !== confirmPassword
    ) {
      setErrorText(
        "Xác nhận mật khẩu mới không khớp."
      );
      return;
    }

    try {
      const data = await apiJson(
        "/api/auth/change-password",
        {
          method: "POST",
          headers: jsonHeaders,
          body: JSON.stringify({
            currentPassword,
            newPassword,
          }),
        }
      );

      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");

      showToast(data.message);
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const createUser = async () => {
    const fullName = window.prompt(
      "Họ tên tài khoản:"
    );
    if (!fullName) return;

    const emailValue = window.prompt(
      "Email:"
    );
    if (!emailValue) return;

    const passwordValue = window.prompt(
      "Mật khẩu (ít nhất 8 ký tự):"
    );
    if (!passwordValue) return;

    const roleValue =
      window.prompt(
        "Quyền: teacher hoặc admin",
        "teacher"
      ) || "teacher";

    try {
      await apiJson(
        "/api/users",
        {
          method: "POST",
          headers: jsonHeaders,
          body: JSON.stringify({
            fullName,
            email: emailValue,
            password: passwordValue,
            role: roleValue.trim(),
          }),
        }
      );

      await loadUsers();
      showToast("Đã tạo tài khoản.");
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const changeUserRole = async (
    item
  ) => {
    const roleValue = window.prompt(
      "Quyền mới: admin hoặc teacher",
      item.role
    );

    if (!roleValue) return;

    try {
      await apiJson(
        `/api/users/${item.id}/role`,
        {
          method: "PUT",
          headers: jsonHeaders,
          body: JSON.stringify({
            role: roleValue.trim(),
          }),
        }
      );

      await loadUsers();
      showToast("Đã cập nhật phân quyền.");
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const editUser = async (
    item
  ) => {
    const fullName = window.prompt(
      "Họ tên:",
      item.full_name
    );

    if (fullName === null) return;

    if (!fullName.trim()) {
      setErrorText(
        "Họ tên không được để trống."
      );
      return;
    }

    const emailValue = window.prompt(
      "Email:",
      item.email
    );

    if (emailValue === null) return;

    if (!emailValue.trim()) {
      setErrorText(
        "Email không được để trống."
      );
      return;
    }

    const passwordValue =
      window.prompt(
        "Mật khẩu mới (để trống nếu giữ nguyên):",
        ""
      );

    if (passwordValue === null) return;

    if (
      passwordValue.length > 0
      && passwordValue.length < 8
    ) {
      setErrorText(
        "Mật khẩu mới phải có ít nhất 8 ký tự."
      );
      return;
    }

    try {
      const updated = await apiJson(
        `/api/users/${item.id}`,
        {
          method: "PUT",
          headers: jsonHeaders,
          body: JSON.stringify({
            fullName: fullName.trim(),
            email: emailValue.trim(),
            password: passwordValue,
          }),
        }
      );

      await loadUsers();

      if (
        Number(item.id) ===
        Number(user?.id)
      ) {
        const nextUser = {
          ...user,
          name: updated.full_name,
          email: updated.email,
          role: updated.role,
        };

        setUser(nextUser);

        localStorage.setItem(
          "user",
          JSON.stringify(nextUser)
        );
      }

      showToast(
        "Đã cập nhật tài khoản."
      );
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const deleteUser = async (
    item
  ) => {
    if (
      Number(item.id) ===
      Number(user?.id)
    ) {
      setErrorText(
        "Không thể tự xóa tài khoản đang đăng nhập."
      );
      return;
    }

    const confirmed =
      window.confirm(
        `Xóa tài khoản "${item.full_name}" (${item.email})?\n\n`
        + "Thao tác này không thể hoàn tác."
      );

    if (!confirmed) return;

    try {
      const data = await apiJson(
        `/api/users/${item.id}`,
        {
          method: "DELETE",
          headers: authHeaders,
        }
      );

      await loadUsers();
      showToast(
        data.message ||
          "Đã xóa tài khoản."
      );
    } catch (error) {
      setErrorText(error.message);
    }
  };

  const exportAlertsCsv = () => {
    const fromTime =
      new Date(
        `${trendFrom}T00:00:00`
      ).getTime();

    const toTime =
      new Date(
        `${trendTo}T23:59:59`
      ).getTime();

    const source =
      allAlerts.filter((item) => {
        const created =
          new Date(
            item.created_at
          ).getTime();

        const classOk =
          !statsClassId
          || String(item.class_id)
            === String(statsClassId);

        const sessionOk =
          !statsSessionId
          || String(item.session_id)
            === String(statsSessionId);

        const studentOk =
          !statsStudentId
          || String(item.student_id)
            === String(statsStudentId);

        const sourceOk =
          !statsStudentId
          || Boolean(item.video_id);

        const timeOk =
          Number.isFinite(created)
          && created >= fromTime
          && created <= toTime;

        return (
          classOk
          && sessionOk
          && studentOk
          && sourceOk
          && timeOk
        );
      });

    const headers = [
      "id",
      "created_at",
      "class_name",
      "session_title",
      "video_name",
      "student_name",
      "student_code",
      "track_id",
      "alert_type",
      "confidence",
      "source_version",
      "note",
    ];

    const escapeCsv = (value) => {
      const text = String(
        value ?? ""
      ).replaceAll('"', '""');

      return `"${text}"`;
    };

    // Excel tren Windows (dac biet voi regional setting Viet Nam)
    // thuong dung dau cham phay lam list separator. Dong sep=; giup Excel
    // tu nhan biet dung dau phan cach khi mo CSV bang double-click.
    // Moi gia tri van duoc quote va escapeCsv() da escape dau ngoac kep,
    // nen dau cham phay ben trong note khong bi tach thanh cot rieng.
    const delimiter = ";";
    const lines = [
      "sep=;",
      headers
        .map((key) => escapeCsv(key))
        .join(delimiter),
      ...source.map((item) =>
        headers
          .map((key) =>
            escapeCsv(item[key])
          )
          .join(delimiter)
      ),
    ];

    const csvContent = "\uFEFF" + lines.join("\r\n");
    const blob = new Blob(
      [csvContent],
      {
        type: "text/csv;charset=utf-8",
      }
    );

    const url =
      URL.createObjectURL(blob);
    const anchor =
      document.createElement("a");

    anchor.href = url;
    anchor.download =
      `bao_cao_canh_bao_${trendFrom}_${trendTo}.csv`;
    anchor.click();

    URL.revokeObjectURL(url);
  };

  if (!token) {
    return (
      <div className="login-page">
        <form
          className="login-card"
          onSubmit={handleLogin}
        >
          <div className="logo">
            KLCN313
          </div>

          <h1>
            Đăng nhập giáo viên
          </h1>

          <p className="subtitle">
            Hệ thống hỗ trợ phát hiện
            dấu hiệu lo âu và mất tập
            trung qua camera.
          </p>

          <label>Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) =>
              setEmail(e.target.value)
            }
            placeholder="Nhập email"
            required
          />

          <label>Mật khẩu</label>
          <input
            type="password"
            value={password}
            onChange={(e) =>
              setPassword(
                e.target.value
              )
            }
            placeholder="Nhập mật khẩu"
            required
          />

          <button
            type="submit"
            disabled={loading}
          >
            {loading
              ? "Đang đăng nhập..."
              : "Đăng nhập"}
          </button>

          {loginMessage && (
            <div className="message error-message">
              {loginMessage}
            </div>
          )}
        </form>
      </div>
    );
  }

  const menuItems = [
    ["overview", "Tổng quan"],
    ["realtime", "Camera / Realtime"],

    ["classes", "Lớp học"],
    ["sessions", "Quản lý buổi học"],
    ["students", "Học sinh"],
    ["alerts", "Cảnh báo"],
    ["history", "Lịch sử AI"],
    ["stats", "Thống kê"],
    ["gradcam", "Grad-CAM"],
    ["videos", "Video"],
    ["account", "Tài khoản"],
  ];

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="sidebar-logo">
          KLCN313
        </div>

        <div className="sidebar-title">
          Teacher Dashboard
        </div>

        <nav className="menu">
          {menuItems.map(
            ([key, label]) => (
              <button
                key={key}
                className={
                  activePage === key
                    ? "menu-active"
                    : ""
                }
                onClick={() => {
                  setActivePage(key);
                  setErrorText("");
                }}
              >
                {label}
              </button>
            )
          )}
        </nav>

        <div className="sidebar-user">
          <strong>
            {user?.name || "Admin"}
          </strong>

          <span>
            {user?.role === "admin"
              ? "Quản trị viên"
              : "Giáo viên"}
          </span>

          <button
            className="logout-button"
            onClick={handleLogout}
          >
            Đăng xuất
          </button>
        </div>
      </aside>

      <main className="dashboard-main">
        {toast && (
          <div className="toast">
            {toast}
          </div>
        )}

        {errorText && (
          <div className="page-error">
            <span>{errorText}</span>
            <button
              onClick={() =>
                setErrorText("")
              }
            >
              ×
            </button>
          </div>
        )}

        <PageErrorBoundary key={activePage}>
        {activePage === "overview" && (
          <OverviewPage
            stats={stats}
            recentAlerts={recentAlerts}
            health={health}
            cameraOnline={cameraOnline}
            formatAlertType={formatAlertType}
            onViewAll={() =>
              setActivePage("alerts")
            }
          />
        )}

        {activePage === "realtime" && (
          <RealtimePage
            token={token}
            tracks={latestTracks}
            cameraOnline={cameraOnline}
            classes={classes}
            sessions={sessions}
            cameraClassId={cameraClassId}
            cameraSessionId={cameraSessionId}
            onCameraClassChange={changeCameraClass}
            onCameraSessionChange={changeCameraSession}
            onCreateSession={() =>
              createSession(
                cameraClassId,
                (id) =>
                  changeCameraSession(id)
              )
            }
            onAssignStudent={assignStudentToTrack}
            formatFusionStatus={
              formatFusionStatus
            }
            fusionClass={fusionClass}
          />
        )}

        {activePage === "classes" && (
          <ClassesPage
            classes={classes}
            isAdmin={canManageClassroom}
            onAdd={addClass}
            onEdit={editClass}
            onDelete={deleteClass}
          />
        )}

        {activePage === "sessions" && (
          <SessionsPage
            classes={classes}
            sessions={sessions}
            isAdmin={isAdmin}
            onCreate={createManagedSession}
            onUpdate={updateManagedSession}
            onDelete={deleteManagedSession}
          />
        )}

        {activePage === "students" && (
          <StudentsPage
            students={students}
            classes={classes}
            selectedClassId={
              selectedClassId
            }
            onSelectedClassId={
              setSelectedClassId
            }
            isAdmin={canManageClassroom}
            onAdd={addStudent}
            onEdit={editStudent}
            onDelete={deleteStudent}
          />
        )}

        {activePage === "alerts" && (
          <AlertsPage
            alerts={filteredAlerts}
            alertFilter={alertFilter}
            classes={classes}
            sessions={sessions}
            classId={alertClassId}
            setClassId={(value) => {
              setAlertClassId(value);
              setAlertSessionId("");
            }}
            sessionId={alertSessionId}
            setSessionId={setAlertSessionId}
            setAlertFilter={
              setAlertFilter
            }
            formatAlertType={
              formatAlertType
            }
            onAssignStudent={assignStudentToTrack}
          />
        )}

        {activePage === "history" && (
          <HistoryPage
            rows={filteredHistory}
            loading={historyLoading}
            loadingMore={historyLoadingMore}
            hasMore={historyHasMore}
            loadError={historyLoadError}
            onLoadMore={loadOlderHistory}
            search={historySearch}
            classes={classes}
            sessions={sessions}
            classId={historyClassId}
            setClassId={(value) => {
              setHistoryClassId(value);
              setHistorySessionId("");
            }}
            sessionId={historySessionId}
            setSessionId={setHistorySessionId}
            setSearch={
              setHistorySearch
            }
            formatFusionStatus={
              formatFusionStatus
            }
            fusionClass={fusionClass}
            onAssignStudent={assignStudentToTrack}
          />
        )}

        {activePage === "stats" && (
          <StatsPage
            trends={trends}
            classes={classes}
            sessions={sessions}
            students={students}
            studentReport={studentReport}
            classId={statsClassId}
            setClassId={(value) => {
              setStatsClassId(value);
              setStatsSessionId("");
              setStatsStudentId("");
              setStudentReport(null);
            }}
            sessionId={statsSessionId}
            setSessionId={setStatsSessionId}
            studentId={statsStudentId}
            setStudentId={(value) => {
              setStatsStudentId(value);
              setStudentReport(null);
            }}
            trendFrom={trendFrom}
            trendTo={trendTo}
            trendPreset={trendPreset}
            setTrendFrom={setTrendFrom}
            setTrendTo={setTrendTo}
            setTrendPreset={setTrendPreset}
            applyTrendPreset={applyTrendPreset}
            onExport={exportAlertsCsv}
          />
        )}

        {activePage === "gradcam" && (
          <GradcamPage
            examples={
              gradcamExamples
            }
          />
        )}

        {activePage === "videos" && (
          <VideosPage
            videos={videos}
            classes={classes}
            sessions={sessions}
            videoClassId={videoClassId}
            setVideoClassId={(value) => {
              setVideoClassId(value);
              setVideoSessionId("");
            }}
            videoSessionId={videoSessionId}
            setVideoSessionId={setVideoSessionId}
            onCreateSession={() =>
              createSession(
                videoClassId,
                setVideoSessionId
              )
            }
            isAdmin={isAdmin}
            fileInputRef={
              fileInputRef
            }
            onUpload={uploadVideo}
            onAnalyze={
              analyzeVideo
            }
            onDelete={deleteVideo}
            onViewResults={(item) => {
              if (item.class_id) {
                setHistoryClassId(String(item.class_id));
              }
              if (item.session_id) {
                setHistorySessionId(String(item.session_id));
              }
              setHistorySearch(
                `video ${item.id}`
              );
              setActivePage("history");
            }}
          />
        )}

        {activePage === "account" && (
          <AccountPage
            user={user}
            currentPassword={
              currentPassword
            }
            setCurrentPassword={
              setCurrentPassword
            }
            newPassword={newPassword}
            setNewPassword={
              setNewPassword
            }
            confirmPassword={
              confirmPassword
            }
            setConfirmPassword={
              setConfirmPassword
            }
            onSubmit={
              changePassword
            }
            isAdmin={isAdmin}
            users={users}
            onCreateUser={createUser}
            onEditUser={editUser}
            onChangeRole={changeUserRole}
            onDeleteUser={deleteUser}
          />
        )}
        </PageErrorBoundary>
      </main>
    </div>
  );
}

function PageHeader({
  title,
  subtitle,
  action,
}) {
  return (
    <header className="dashboard-header">
      <div>
        <h1>{title}</h1>
        {subtitle && (
          <p>{subtitle}</p>
        )}
      </div>

      {action}
    </header>
  );
}

function OverviewPage({
  stats,
  recentAlerts,
  health,
  cameraOnline,
  formatAlertType,
  onViewAll,
}) {
  return (
    <>
      <PageHeader
        title="Tổng quan"
        subtitle="Hệ thống giám sát trạng thái lớp học"
        action={
          <div
            className={
              health.backend
                ? "backend-status"
                : "backend-status offline"
            }
          >
            ●{" "}
            {health.backend
              ? "Hệ thống đang hoạt động"
              : "Backend mất kết nối"}
          </div>
        }
      />

      <section className="overview-grid">
        <StatCard
          label="Lớp học"
          value={stats.total_classes}
          note="Tổng số lớp"
        />

        <StatCard
          label="Học sinh"
          value={stats.total_students}
          note="Tổng số học sinh"
        />

        <StatCard
          label="Mất tập trung"
          value={stats.distracted}
          note="Đang được theo dõi"
          className="warning-card"
        />

        <StatCard
          label="Dấu hiệu lo âu"
          value={stats.anxiety}
          note="Đang được theo dõi"
          className="anxiety-card"
        />
      </section>

      <section className="dashboard-section">
        <div className="section-header">
          <div>
            <h2>
              Trạng thái lớp học
            </h2>
            <p>
              Theo dõi các đối tượng
              đang được hệ thống nhận diện
            </p>
          </div>
        </div>

        <div className="status-summary-grid">
          <MiniStat
            label="Track đang theo dõi"
            value={stats.active_tracks}
          />
          <MiniStat
            label="Cảnh báo 24 giờ"
            value={stats.alerts_24h}
          />
          <MiniStat
            label="Video đã quản lý"
            value={stats.total_videos}
          />
          <MiniStat
            label="Camera AI"
            value={
              cameraOnline
                ? "Đang chạy"
                : "Chưa chạy"
            }
          />
        </div>
      </section>

      <section className="bottom-grid">
        <div className="bottom-card">
          <div className="bottom-card-header">
            <div>
              <h2>
                Cảnh báo gần đây
              </h2>
              <p>
                Các cảnh báo mới nhất
                từ hệ thống
              </p>
            </div>

            <button
              className="view-all-button"
              onClick={onViewAll}
            >
              Xem tất cả
            </button>
          </div>

          {recentAlerts.length === 0 ? (
            <EmptyState
              title="Chưa có cảnh báo mới"
              text="Cảnh báo mất tập trung hoặc dấu hiệu lo âu sẽ xuất hiện tại đây."
            />
          ) : (
            <div className="recent-alert-list">
              {recentAlerts.map(
                (alert) => (
                  <AlertRow
                    key={alert.id}
                    alert={alert}
                    formatAlertType={
                      formatAlertType
                    }
                  />
                )
              )}
            </div>
          )}
        </div>

        <div className="bottom-card">
          <div className="bottom-card-header">
            <div>
              <h2>
                Hoạt động hệ thống
              </h2>
              <p>
                Trạng thái các thành phần chính
              </p>
            </div>
          </div>

          <SystemRow
            name="Backend API"
            detail="Node.js / Express"
            ok={health.backend}
            okText="Hoạt động"
            failText="Mất kết nối"
          />

          <SystemRow
            name="Cơ sở dữ liệu"
            detail="PostgreSQL"
            ok={health.database}
            okText="Kết nối"
            failText="Mất kết nối"
          />

          <SystemRow
            name="Camera AI"
            detail="V14 Multi-Person"
            ok={cameraOnline}
            okText="Đang chạy"
            failText="Chưa chạy"
          />

          <SystemRow
            name="Web Dashboard"
            detail="React / Vite"
            ok
            okText="Hoạt động"
            failText=""
          />
        </div>
      </section>
    </>
  );
}

function StatCard({
  label,
  value,
  note,
  className = "",
}) {
  return (
    <div
      className={`overview-card ${className}`}
    >
      <div className="overview-label">
        {label}
      </div>

      <div className="overview-number">
        {value ?? 0}
      </div>

      <div className="overview-note">
        {note}
      </div>
    </div>
  );
}

function MiniStat({
  label,
  value,
}) {
  return (
    <div className="mini-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SystemRow({
  name,
  detail,
  ok,
  okText,
  failText,
}) {
  return (
    <div className="system-item">
      <div className="system-left">
        <span
          className={
            ok
              ? "status-dot online-dot"
              : "status-dot waiting-dot"
          }
        />

        <div>
          <strong>{name}</strong>
          <p>{detail}</p>
        </div>
      </div>

      <span
        className={
          ok
            ? "system-status online-text"
            : "system-status waiting-text"
        }
      >
        {ok ? okText : failText}
      </span>
    </div>
  );
}

function RealtimePage({
  token,
  tracks,
  cameraOnline,
  classes,
  sessions,
  cameraClassId,
  cameraSessionId,
  onCameraClassChange,
  onCameraSessionChange,
  onCreateSession,
  onAssignStudent,
  formatFusionStatus,
  fusionClass,
}) {
  const sessionOptions =
    sessions.filter(
      (item) =>
        String(item.class_id)
        === String(cameraClassId)
    );

  return (
    <>
      <PageHeader
        title="Camera / Realtime"
        subtitle="Theo dõi AI theo đúng Lớp học và Buổi học"
        action={
          <div
            className={
              cameraOnline
                ? "backend-status"
                : "backend-status offline"
            }
          >
            ●{" "}
            {cameraOnline
              ? "Camera AI đang gửi dữ liệu"
              : "Chưa có dữ liệu AI mới"}
          </div>
        }
      />

      <section className="dashboard-section">
        <div className="class-context-grid">
          <label>
            <span>Lớp học</span>
            <select
              value={cameraClassId}
              onChange={(e) =>
                onCameraClassChange(
                  e.target.value
                )
              }
            >
              <option value="">
                -- Chọn lớp --
              </option>
              {classes.map((item) => (
                <option
                  key={item.id}
                  value={item.id}
                >
                  {item.name}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Buổi học</span>
            <select
              value={cameraSessionId}
              disabled={!cameraClassId}
              onChange={(e) =>
                onCameraSessionChange(
                  e.target.value
                )
              }
            >
              <option value="">
                -- Chọn buổi học --
              </option>
              {sessionOptions.map(
                (item) => (
                  <option
                    key={item.id}
                    value={item.id}
                  >
                    {item.title}
                    {" • "}
                    {new Date(
                      `${String(item.session_date).slice(0,10)}T00:00:00`
                    ).toLocaleDateString(
                      "vi-VN"
                    )}
                  </option>
                )
              )}
            </select>
          </label>

          <button
            className="secondary-button session-create-button"
            disabled={!cameraClassId}
            onClick={onCreateSession}
          >
            + Tạo buổi học
          </button>
        </div>

        <p className="info-note">
          Track ID chỉ là định danh tạm thời. Khi biết Track thuộc học sinh nào,
          bấm “Gán học sinh” để lưu đúng hồ sơ. Kết quả Camera được lưu vào PostgreSQL
          và xem lại trong mục “Lịch sử AI”. Bảng Realtime chỉ giữ Track có dữ liệu mới
          trong 5 phút gần nhất; Track cũ biến mất khỏi bảng live nhưng không vì thế mà
          bị xóa khỏi lịch sử đã lưu.
        </p>

        {!cameraClassId ? (
          <EmptyState
            title="Chưa chọn lớp"
            text="Hãy chọn lớp học trước."
          />
        ) : !cameraSessionId ? (
          <EmptyState
            title="Chưa chọn buổi học"
            text="Chọn hoặc tạo buổi học trước khi chạy Camera AI."
          />
        ) : tracks.length === 0 ? (
          <EmptyState
            title="Chưa có Track realtime"
            text="Đã chọn Lớp + Buổi học. Bây giờ chạy V14 Camera."
          />
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Track</th>
                  <th>Học sinh</th>
                  <th>Trạng thái</th>
                  <th>Lý do</th>
                  <th>FER</th>
                  <th>Engagement</th>
                  <th>Cập nhật</th>
                </tr>
              </thead>
              <tbody>
                {tracks.map((item) => (
                  <tr key={item.track_id}>
                    <td>
                      <strong>
                        #{item.track_id}
                      </strong>
                    </td>
                    <td>
                      {item.student_name ? (
                        <>
                          <strong>
                            {item.student_name}
                          </strong>
                          {item.student_code && (
                            <div className="muted-small">
                              {item.student_code}
                            </div>
                          )}
                        </>
                      ) : (
                        <button
                          className="tiny-button"
                          onClick={() =>
                            onAssignStudent(item)
                          }
                        >
                          Gán học sinh
                        </button>
                      )}
                    </td>
                    <td>
                      <span
                        className={fusionClass(
                          item.fusion_status
                        )}
                      >
                        {formatFusionStatus(
                          item.fusion_status
                        )}
                      </span>
                    </td>
                    <td>
                      {item.fusion_reason || "-"}
                    </td>
                    <td>
                      {item.emotion || "-"}
                    </td>
                    <td>
                      {item.engagement || "-"}
                    </td>
                    <td>
                      {new Date(
                        item.captured_at
                        || item.created_at
                      ).toLocaleTimeString(
                        "vi-VN"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <CameraRecordings classId={cameraClassId} sessionId={cameraSessionId} token={token} />
      </section>
    </>
  );
}

function ClassesPage({
  classes,
  isAdmin,
  onAdd,
  onEdit,
  onDelete,
}) {
  return (
    <>
      <PageHeader
        title="Lớp học"
        subtitle="Giáo viên và quản trị viên có thể thêm, sửa, xóa lớp học"
        action={
          isAdmin ? (
            <button
              className="primary-button"
              onClick={onAdd}
            >
              + Thêm lớp
            </button>
          ) : null
        }
      />

      <section className="dashboard-section">
        {classes.length === 0 ? (
          <EmptyState
            title="Chưa có lớp học"
            text="Tạo lớp đầu tiên để quản lý học sinh."
          />
        ) : (
          <div className="card-grid">
            {classes.map((item) => (
              <article
                className="class-card"
                key={item.id}
              >
                <div>
                  <span className="small-muted">
                    Lớp #{item.id}
                  </span>
                  <h3>{item.name}</h3>
                  <p>
                    {item.description ||
                      "Không có mô tả"}
                  </p>
                </div>

                <div className="class-card-footer">
                  <span>
                    {item.student_count ?? 0} học sinh
                  </span>

                  {isAdmin && (
                    <div className="row-actions">
                      <button
                        onClick={() =>
                          onEdit(item)
                        }
                      >
                        Sửa
                      </button>
                      <button
                        className="danger-link"
                        onClick={() =>
                          onDelete(item)
                        }
                      >
                        Xóa
                      </button>
                    </div>
                  )}
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </>
  );
}

function StudentsPage({
  students,
  classes,
  selectedClassId,
  onSelectedClassId,
  isAdmin,
  onAdd,
  onEdit,
  onDelete,
}) {
  const visible = selectedClassId
    ? students.filter(
        (item) =>
          String(item.class_id) ===
          String(selectedClassId)
      )
    : students;

  return (
    <>
      <PageHeader
        title="Học sinh"
        subtitle="Giáo viên và quản trị viên có thể quản lý danh sách học sinh theo lớp"
        action={
          isAdmin ? (
            <button
              className="primary-button"
              onClick={onAdd}
              disabled={classes.length === 0}
              title={
                classes.length === 0
                  ? "Hãy tạo lớp học trước."
                  : "Thêm học sinh vào lớp đang chọn."
              }
            >
              + Thêm học sinh
            </button>
          ) : null
        }
      />

      <section className="dashboard-section">
        <div className="toolbar">
          <label>
            Lọc theo lớp
            <select
              value={selectedClassId}
              onChange={(e) =>
                onSelectedClassId(
                  e.target.value
                )
              }
            >
              <option value="">
                Tất cả lớp
              </option>
              {classes.map((item) => (
                <option
                  value={item.id}
                  key={item.id}
                >
                  {item.name}
                </option>
              ))}
            </select>
          </label>
        </div>

        {visible.length === 0 ? (
          <EmptyState
            title="Chưa có học sinh"
            text="Thêm học sinh vào lớp để bắt đầu quản lý."
          />
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Mã</th>
                  <th>Họ tên</th>
                  <th>Lớp</th>
                  <th>Ngày tạo</th>
                  {isAdmin && (
                    <th>Thao tác</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {visible.map((item) => (
                  <tr key={item.id}>
                    <td>
                      {item.student_code ||
                        "-"}
                    </td>
                    <td>
                      <strong>
                        {item.full_name}
                      </strong>
                    </td>
                    <td>
                      {item.class_name ||
                        classes.find(
                          (c) =>
                            c.id ===
                            item.class_id
                        )?.name ||
                        "-"}
                    </td>
                    <td>
                      {new Date(
                        item.created_at
                      ).toLocaleDateString(
                        "vi-VN"
                      )}
                    </td>
                    {isAdmin && (
                      <td>
                        <div className="row-actions">
                          <button
                            onClick={() =>
                              onEdit(item)
                            }
                          >
                            Sửa
                          </button>
                          <button
                            className="danger-link"
                            onClick={() =>
                              onDelete(item)
                            }
                          >
                            Xóa
                          </button>
                        </div>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}

function SessionsPage({
  classes,
  sessions,
  isAdmin,
  onCreate,
  onUpdate,
  onDelete,
}) {
  const [classFilterId, setClassFilterId] = useState("");
  const [formOpen, setFormOpen] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [saving, setSaving] = useState(false);
  const blankForm = (classId = "") => ({
    classId,
    title: "",
    sessionDate: toDateInput(new Date()),
    startTime: "",
    endTime: "",
    note: "",
  });
  const [form, setForm] = useState(() => blankForm(""));

  useEffect(() => {
    if (!classes.length) {
      setClassFilterId("");
      return;
    }
    if (
      classFilterId
      && !classes.some((item) => String(item.id) === String(classFilterId))
    ) {
      setClassFilterId("");
    }
  }, [classes, classFilterId]);

  const rows = sessions
    .filter((item) => !classFilterId || String(item.class_id) === String(classFilterId))
    .slice()
    .sort((a, b) => {
      const classCompare = String(a.class_name || "").localeCompare(String(b.class_name || ""), "vi");
      if (classCompare) return classCompare;
      const dateCompare = String(b.session_date || "").localeCompare(String(a.session_date || ""));
      return dateCompare || Number(b.id) - Number(a.id);
    });

  const openCreate = () => {
    setEditingId(null);
    setForm(blankForm(classFilterId || String(classes[0]?.id || "")));
    setFormOpen(true);
  };

  const openEdit = (item) => {
    setEditingId(String(item.id));
    setForm({
      classId: String(item.class_id),
      title: item.title || "",
      sessionDate: String(item.session_date || "").slice(0, 10),
      startTime: String(item.start_time || "").slice(0, 5),
      endTime: String(item.end_time || "").slice(0, 5),
      note: item.note || "",
    });
    setClassFilterId(String(item.class_id));
    setFormOpen(true);
  };

  const submitForm = async (event) => {
    event.preventDefault();
    if (!form.classId) return;
    setSaving(true);
    try {
      const success = editingId
        ? await onUpdate(editingId, form)
        : await onCreate(form);
      if (success) {
        setFormOpen(false);
        setEditingId(null);
      }
    } finally {
      setSaving(false);
    }
  };

  const requestDelete = async (item) => {
    if (!isAdmin) return;
    const className = item.class_name
      || classes.find((c) => String(c.id) === String(item.class_id))?.name
      || `Lớp #${item.class_id}`;
    const confirmed = window.confirm(
      `Bạn chắc chắn muốn xóa “${item.title}” của ${className} (${String(item.session_date || "").slice(0, 10)})?\n\nCác bản ghi lịch sử AI/cảnh báo vẫn còn trong cơ sở dữ liệu nhưng sẽ mất liên kết với buổi này. Video đã gắn buổi cũng sẽ không còn gắn buổi; các liên kết Track–học sinh thuộc buổi này sẽ bị xóa. Thao tác này không thể hoàn tác.`
    );
    if (confirmed) await onDelete(item);
  };

  return (
    <>
      <PageHeader
        title="Quản lý buổi học"
        subtitle="Thêm, chỉnh sửa và xóa buổi học; mỗi buổi luôn hiển thị kèm lớp, ngày học và mã buổi để tránh nhầm Buổi 2 giữa các lớp."
        action={(
          <button className="primary-button" onClick={openCreate} disabled={classes.length === 0}>
            + Thêm buổi học
          </button>
        )}
      />

      <section className="dashboard-section">
        <div className="session-manager-toolbar">
          <label>
            <span>Lọc theo lớp</span>
            <select value={classFilterId} onChange={(event) => setClassFilterId(event.target.value)}>
              <option value="">Tất cả lớp học</option>
              {classes.map((item) => (
                <option key={item.id} value={item.id}>{item.name}</option>
              ))}
            </select>
          </label>
          <div className="session-manager-count">{rows.length} buổi học</div>
        </div>

        {formOpen && (
          <form className="session-editor" onSubmit={submitForm}>
            <div className="session-editor-heading">
              <h3>{editingId ? `Chỉnh sửa buổi #${editingId}` : "Thêm buổi học mới"}</h3>
              <button type="button" className="secondary-button" onClick={() => setFormOpen(false)}>Đóng</button>
            </div>
            <div className="session-editor-grid">
              <label>
                <span>Lớp học *</span>
                <select required value={form.classId} disabled={Boolean(editingId)} onChange={(event) => setForm((current) => ({ ...current, classId: event.target.value }))}>
                  <option value="">-- Chọn lớp --</option>
                  {classes.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
                </select>
              </label>
              <label>
                <span>Tên buổi học *</span>
                <input required maxLength={160} value={form.title} placeholder="Ví dụ: Buổi 3 - Chương 4" onChange={(event) => setForm((current) => ({ ...current, title: event.target.value }))} />
              </label>
              <label>
                <span>Ngày học *</span>
                <input required type="date" value={form.sessionDate} onChange={(event) => setForm((current) => ({ ...current, sessionDate: event.target.value }))} />
              </label>
              <label>
                <span>Giờ bắt đầu</span>
                <input type="time" value={form.startTime} onChange={(event) => setForm((current) => ({ ...current, startTime: event.target.value }))} />
              </label>
              <label>
                <span>Giờ kết thúc</span>
                <input type="time" value={form.endTime} onChange={(event) => setForm((current) => ({ ...current, endTime: event.target.value }))} />
              </label>
              <label className="session-note-field">
                <span>Ghi chú</span>
                <input value={form.note} maxLength={1000} placeholder="Ví dụ: Chương 4, tiết thực hành..." onChange={(event) => setForm((current) => ({ ...current, note: event.target.value }))} />
              </label>
            </div>
            <div className="session-editor-actions">
              <button className="primary-button" type="submit" disabled={saving || !form.classId || !form.title.trim() || !form.sessionDate}>
                {saving ? "Đang lưu..." : editingId ? "Lưu thay đổi" : "Tạo buổi học"}
              </button>
              <button className="secondary-button" type="button" onClick={() => setFormOpen(false)} disabled={saving}>Hủy</button>
            </div>
          </form>
        )}

        {classes.length === 0 ? (
          <EmptyState title="Chưa có lớp học" text="Hãy tạo lớp học trước khi thêm buổi học." />
        ) : rows.length === 0 ? (
          <EmptyState title="Chưa có buổi học" text="Lớp đang chọn chưa có buổi học. Bấm “Thêm buổi học” để tạo mới." />
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Mã buổi</th>
                  <th>Lớp học</th>
                  <th>Tên buổi</th>
                  <th>Ngày học</th>
                  <th>Thời gian</th>
                  <th>Video</th>
                  <th>Cảnh báo</th>
                  <th>Thao tác</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((item) => {
                  const className = item.class_name
                    || classes.find((c) => String(c.id) === String(item.class_id))?.name
                    || `Lớp #${item.class_id}`;
                  return (
                    <tr key={item.id}>
                      <td><strong>#{item.id}</strong></td>
                      <td>{className}</td>
                      <td>
                        <strong>{item.title}</strong>
                        {item.note && <div className="muted-small">{item.note}</div>}
                      </td>
                      <td>{String(item.session_date || "").slice(0, 10)}</td>
                      <td>{item.start_time ? String(item.start_time).slice(0, 5) : "—"}{item.end_time ? ` – ${String(item.end_time).slice(0, 5)}` : ""}</td>
                      <td>{Number(item.video_count || 0)}</td>
                      <td>{Number(item.alert_count || 0)}</td>
                      <td>
                        <div className="row-actions">
                          <button className="secondary-button" onClick={() => openEdit(item)}>Sửa</button>
                          <button className="danger-button" disabled={!isAdmin} title={isAdmin ? "Xóa buổi học" : "Chỉ quản trị viên mới được xóa buổi học"} onClick={() => requestDelete(item)}>Xóa</button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        {!isAdmin && <p className="info-note">Tài khoản giáo viên có thể thêm và sửa buổi học. Chỉ quản trị viên mới được xóa buổi học.</p>}
      </section>
    </>
  );
}

function AlertsPage({
  alerts,
  alertFilter,
  setAlertFilter,
  classes,
  sessions,
  classId,
  setClassId,
  sessionId,
  setSessionId,
  formatAlertType,
  onAssignStudent,
}) {
  const sessionOptions =
    sessions.filter(
      (item) =>
        !classId
        || String(item.class_id)
          === String(classId)
    );

  return (
    <>
      <PageHeader
        title="Cảnh báo"
        subtitle="Cảnh báo theo Video / Buổi học / Lớp / Học sinh"
      />

      <section className="dashboard-section">
        <div className="toolbar toolbar-between">
          <div className="filter-group">
            <label>
              Lớp học
              <select
                value={classId}
                onChange={(e) =>
                  setClassId(
                    e.target.value
                  )
                }
              >
                <option value="">
                  Tất cả lớp
                </option>
                {classes.map(
                  (item) => (
                    <option
                      key={item.id}
                      value={item.id}
                    >
                      {item.name}
                    </option>
                  )
                )}
              </select>
            </label>

            <label>
              Buổi học
              <select
                value={sessionId}
                onChange={(e) =>
                  setSessionId(
                    e.target.value
                  )
                }
              >
                <option value="">
                  Tất cả buổi học
                </option>
                {sessionOptions.map(
                  (item) => (
                    <option
                      key={item.id}
                      value={item.id}
                    >
                      {item.title}
                    </option>
                  )
                )}
              </select>
            </label>
          </div>

          <div className="filter-buttons">
            <button
              className={
                alertFilter === "all"
                  ? "filter-active"
                  : ""
              }
              onClick={() =>
                setAlertFilter("all")
              }
            >
              Tất cả
            </button>

            <button
              className={
                alertFilter ===
                "mat_tap_trung"
                  ? "filter-active"
                  : ""
              }
              onClick={() =>
                setAlertFilter(
                  "mat_tap_trung"
                )
              }
            >
              Mất tập trung
            </button>

            <button
              className={
                alertFilter ===
                "dau_hieu_lo_au"
                  ? "filter-active"
                  : ""
              }
              onClick={() =>
                setAlertFilter(
                  "dau_hieu_lo_au"
                )
              }
            >
              Dấu hiệu lo âu
            </button>
          </div>
        </div>

        {alerts.length === 0 ? (
          <EmptyState
            title="Không có cảnh báo"
            text="Không có dữ liệu phù hợp với bộ lọc hiện tại."
          />
        ) : (
          <div className="alert-page-list">
            {alerts.map((alert) => (
              <AlertRow
                key={alert.id}
                alert={alert}
                formatAlertType={
                  formatAlertType
                }
                onAssignStudent={
                  onAssignStudent
                }
                detailed
              />
            ))}
          </div>
        )}
      </section>
    </>
  );
}

function AlertRow({
  alert,
  formatAlertType,
  onAssignStudent,
  detailed = false,
}) {
  return (
    <div className="recent-alert-item alert-rich-row">
      <div
        className={
          alert.alert_type ===
          "dau_hieu_lo_au"
            ? "alert-dot anxiety-dot"
            : "alert-dot distracted-dot"
        }
      />

      <div className="recent-alert-content">
        <div className="recent-alert-row">
          <strong>
            {formatAlertType(
              alert.alert_type
            )}
          </strong>

          <span>
            {new Date(
              alert.created_at
            ).toLocaleString("vi-VN")}
          </span>
        </div>

        <div className="context-chip-row">
          <span className="context-chip">
            Lớp: {alert.class_name || "-"}
          </span>
          <span className="context-chip">
            Buổi: {alert.session_title || "-"}
          </span>
          <span className="context-chip">
            Video: {alert.video_name || "Camera"}
          </span>
        </div>

        <div className="student-context-line">
          <strong>Học sinh: </strong>

          {alert.student_name ? (
            <span>
              {alert.student_name}
              {alert.student_code
                ? ` • ${alert.student_code}`
                : ""}
            </span>
          ) : (
            <>
              <span>
                Chưa gán học sinh
                {alert.track_id !== null &&
                alert.track_id !== undefined
                  ? ` • Track ${alert.track_id}`
                  : ""}
              </span>

              {onAssignStudent &&
                alert.session_id &&
                alert.track_id !== null &&
                alert.track_id !== undefined && (
                  <button
                    className="tiny-button"
                    onClick={() =>
                      onAssignStudent(alert)
                    }
                  >
                    Gán học sinh
                  </button>
                )}
            </>
          )}
        </div>

        {alert.gradcam_url && (
          <div className="record-gradcam">
            <span>Grad-CAM minh họa</span>
            <ProtectedMedia path={alert.gradcam_url} alt="Grad-CAM minh họa" />
          </div>
        )}

        {detailed && (
          <div className="alert-details">
            <span>
              Confidence:{" "}
              {alert.confidence === null ||
              alert.confidence === undefined
                ? "-"
                : Number(
                    alert.confidence
                  ).toFixed(2)}
            </span>
            <span>
              Nguồn:{" "}
              {alert.source_version ||
                "-"}
            </span>
            {alert.note && (
              <p>{alert.note}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function HistoryPage({
  rows,
  loading,
  loadingMore,
  hasMore,
  loadError,
  onLoadMore,
  search,
  setSearch,
  classes,
  sessions,
  classId,
  setClassId,
  sessionId,
  setSessionId,
  formatFusionStatus,
  fusionClass,
  onAssignStudent,
}) {
  const sessionOptions =
    sessions.filter(
      (item) =>
        !classId
        || String(item.class_id)
          === String(classId)
    );

  return (
    <>
      <PageHeader
        title="Lịch sử AI"
        subtitle="Kết quả AI theo Video / Buổi học / Lớp / Học sinh"
      />

      <section className="dashboard-section">
        <div className="toolbar toolbar-between">
          <div className="filter-group">
            <label>
              Lớp học
              <select
                value={classId}
                onChange={(e) =>
                  setClassId(
                    e.target.value
                  )
                }
              >
                <option value="">
                  Tất cả lớp
                </option>
                {classes.map(
                  (item) => (
                    <option
                      key={item.id}
                      value={item.id}
                    >
                      {item.name}
                    </option>
                  )
                )}
              </select>
            </label>

            <label>
              Buổi học
              <select
                value={sessionId}
                onChange={(e) =>
                  setSessionId(
                    e.target.value
                  )
                }
              >
                <option value="">
                  Tất cả buổi học
                </option>
                {sessionOptions.map(
                  (item) => (
                    <option
                      key={item.id}
                      value={item.id}
                    >
                      {item.title}
                    </option>
                  )
                )}
              </select>
            </label>
          </div>

          <input
            className="search-input"
            placeholder="Tìm học sinh, Track, Video, Fusion, FER..."
            value={search}
            onChange={(e) =>
              setSearch(e.target.value)
            }
          />
        </div>

        {loading && rows.length === 0 ? (
          <EmptyState
            title="Đang tải lịch sử AI"
            text="Đang đọc dữ liệu đã lưu từ Backend/PostgreSQL."
          />
        ) : loadError && rows.length === 0 ? (
          <EmptyState
            title="Không tải được lịch sử AI"
            text={loadError}
          />
        ) : rows.length === 0 ? (
          <EmptyState
            title="Chưa có dữ liệu AI phù hợp"
            text="Không có bản ghi khớp bộ lọc hiện tại. Hãy kiểm tra lớp, buổi học hoặc video đã chọn."
          />
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Thời gian</th>
                  <th>Lớp</th>
                  <th>Buổi học</th>
                  <th>Nguồn</th>
                  <th>Học sinh</th>
                  <th>Track</th>
                  <th>Fusion</th>
                  <th>FER</th>
                  <th>Engagement</th>
                  <th>Grad-CAM</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((item) => (
                  <tr key={item.id}>
                    <td>
                      {new Date(
                        item.created_at
                      ).toLocaleString(
                        "vi-VN"
                      )}
                    </td>

                    <td>
                      {item.class_name || "-"}
                    </td>

                    <td>
                      {item.session_title || "-"}
                    </td>

                    <td>
                      {item.video_name
                        || (item.video_id
                          ? `Video #${item.video_id}`
                          : "Camera")}
                    </td>

                    <td>
                      {item.student_name ? (
                        <>
                          <strong>
                            {item.student_name}
                          </strong>
                          {item.student_code && (
                            <div className="muted-small">
                              {item.student_code}
                            </div>
                          )}
                        </>
                      ) : (
                        <div className="student-unmapped">
                          <span>
                            Chưa gán học sinh
                          </span>

                          {item.session_id &&
                            item.track_id !== null &&
                            item.track_id !== undefined && (
                              <button
                                className="tiny-button"
                                onClick={() =>
                                  onAssignStudent(item)
                                }
                              >
                                Gán
                              </button>
                            )}
                        </div>
                      )}
                    </td>

                    <td>
                      {item.track_id ?? "-"}
                    </td>

                    <td>
                      <span
                        className={fusionClass(
                          item.fusion_status
                        )}
                        title={
                          item.fusion_reason
                          || ""
                        }
                      >
                        {formatFusionStatus(
                          item.fusion_status
                        )}
                      </span>
                    </td>

                    <td>
                      {item.emotion || "-"}
                    </td>

                    <td>
                      {item.engagement || "-"}
                    </td>

                    <td>
                      {item.gradcam_url ? (
                        <ProtectedMedia path={item.gradcam_url} alt="Grad-CAM" className="gradcam-thumb" openOnClick />
                      ) : (
                        "-"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {rows.length > 0 && (
          <p className="muted-small">
            Đang hiển thị {rows.length} bản ghi phù hợp trong số bản ghi đã tải.
            Dữ liệu được lưu ở PostgreSQL; bấm tải bản ghi cũ hơn để xem lịch sử xa hơn.
          </p>
        )}

        {hasMore && (
          <div className="toolbar toolbar-center">
            <button
              className="secondary-button"
              onClick={onLoadMore}
              disabled={loadingMore || loading}
            >
              {loadingMore ? "Đang tải bản ghi cũ hơn..." : "Tải bản ghi cũ hơn"}
            </button>
          </div>
        )}
      </section>
    </>
  );
}

function StatsPage({
  trends,
  classes,
  sessions,
  students,
  studentReport,
  classId,
  setClassId,
  sessionId,
  setSessionId,
  studentId,
  setStudentId,
  trendFrom,
  trendTo,
  trendPreset,
  setTrendFrom,
  setTrendTo,
  setTrendPreset,
  applyTrendPreset,
  onExport,
}) {
  const sessionOptions =
    sessions.filter(
      (item) =>
        !classId
        || String(item.class_id)
          === String(classId)
    );

  const studentOptions =
    students.filter(
      (item) =>
        classId
        && String(item.class_id)
          === String(classId)
    );

  const selectedStudent =
    studentOptions.find(
      (item) =>
        String(item.id) === String(studentId)
    );

  const max = Math.max(
    ...trends.map(
      (item) => Number(item.total) || 0
    ),
    1
  );

  const totalAlerts = trends.reduce(
    (sum, item) =>
      sum + Number(item.total || 0),
    0
  );

  const distractedAlerts = trends.reduce(
    (sum, item) =>
      sum + Number(item.distracted || 0),
    0
  );

  const anxietyAlerts = trends.reduce(
    (sum, item) =>
      sum + Number(item.anxiety || 0),
    0
  );

  const daysWithAlerts = trends.filter(
    (item) => Number(item.total) > 0
  ).length;

  const rangeLabel =
    trendFrom && trendTo
      ? `${new Date(
          `${trendFrom}T00:00:00`
        ).toLocaleDateString("vi-VN")} → ${new Date(
          `${trendTo}T00:00:00`
        ).toLocaleDateString("vi-VN")}`
      : "Chưa chọn khoảng thời gian";

  return (
    <>
      <PageHeader
        title="Báo cáo - Thống kê"
        subtitle="Lọc theo lớp hoặc xem báo cáo chi tiết từng học sinh từ video đã phân tích"
        action={
          <button
            className="primary-button"
            onClick={onExport}
          >
            Xuất CSV cảnh báo
          </button>
        }
      />

      <section className="dashboard-section history-filter-card">
        <div className="section-header">
          <div>
            <h2>Khoảng thời gian</h2>
            <p>
              Chọn nhanh hoặc chọn chính xác từ ngày đến ngày để xem lại dữ liệu cũ.
            </p>
          </div>
        </div>

        <div className="stats-class-filter filter-group">
          <label>
            Lớp cần xem
            <select
              value={classId}
              onChange={(e) =>
                setClassId(
                  e.target.value
                )
              }
            >
              <option value="">
                Tất cả lớp
              </option>
              {classes.map((item) => (
                <option
                  key={item.id}
                  value={item.id}
                >
                  {item.name}
                </option>
              ))}
            </select>
          </label>

          <label>
            Buổi học
            <select
              value={sessionId}
              onChange={(e) =>
                setSessionId(
                  e.target.value
                )
              }
            >
              <option value="">
                Tất cả buổi học
              </option>
              {sessionOptions.map(
                (item) => (
                  <option
                    key={item.id}
                    value={item.id}
                  >
                    {item.title}
                  </option>
                )
              )}
            </select>
          </label>

          <label>
            Học sinh
            <select
              value={studentId}
              disabled={!classId}
              onChange={(e) =>
                setStudentId(
                  e.target.value
                )
              }
            >
              <option value="">
                {classId
                  ? "Tất cả học sinh"
                  : "Chọn lớp trước"}
              </option>
              {studentOptions.map(
                (item) => (
                  <option
                    key={item.id}
                    value={item.id}
                  >
                    {item.student_code
                      ? `${item.student_code} - ${item.full_name}`
                      : item.full_name}
                  </option>
                )
              )}
            </select>
          </label>
        </div>

        {studentId && (
          <div className="student-report-source-note">
            Báo cáo học sinh chỉ tính dữ liệu từ
            <strong> video đã phân tích </strong>
            mà Track đã được gán cho học sinh này.
          </div>
        )}

        <div className="history-presets">
          {[
            ["today", "Hôm nay"],
            ["7", "7 ngày"],
            ["30", "30 ngày"],
            ["month", "Tháng này"],
            ["prev-month", "Tháng trước"],
            ["all", "Toàn bộ"],
          ].map(([value, label]) => (
            <button
              key={value}
              className={
                trendPreset === value
                  ? "history-preset-active"
                  : ""
              }
              onClick={() =>
                applyTrendPreset(value)
              }
            >
              {label}
            </button>
          ))}
        </div>

        <div className="date-range-row">
          <label>
            <span>Từ ngày</span>
            <input
              type="date"
              value={trendFrom}
              max={trendTo}
              onChange={(e) => {
                setTrendPreset("custom");
                setTrendFrom(e.target.value);
              }}
            />
          </label>

          <div className="date-range-arrow">
            →
          </div>

          <label>
            <span>Đến ngày</span>
            <input
              type="date"
              value={trendTo}
              min={trendFrom}
              max={toDateInput(new Date())}
              onChange={(e) => {
                setTrendPreset("custom");
                setTrendTo(e.target.value);
              }}
            />
          </label>

          <div className="range-summary">
            <strong>Đang xem</strong>
            <span>{rangeLabel}</span>
          </div>
        </div>
      </section>

      <section className="overview-grid stats-cards">
        <StatCard
          label="Tổng cảnh báo"
          value={totalAlerts}
          note="Trong khoảng đang chọn"
        />
        <StatCard
          label="Mất tập trung"
          value={distractedAlerts}
          note="Trong khoảng đang chọn"
          className="warning-card"
        />
        <StatCard
          label="Dấu hiệu lo âu"
          value={anxietyAlerts}
          note="Trong khoảng đang chọn"
          className="anxiety-card"
        />
        <StatCard
          label="Ngày có cảnh báo"
          value={daysWithAlerts}
          note={`${trends.length} ngày được thống kê`}
        />
      </section>

      {studentId && (
        <section className="dashboard-section student-detail-report">
          <div className="section-header student-report-heading">
            <div>
              <h2>Báo cáo chi tiết học sinh</h2>
              <p>
                {selectedStudent
                  ? `${selectedStudent.student_code || ""} ${selectedStudent.full_name}`.trim()
                  : "Học sinh đang chọn"}
                {studentReport?.student?.class_name
                  ? ` • ${studentReport.student.class_name}`
                  : ""}
              </p>
            </div>

            <span className="student-video-badge">
              Dữ liệu từ video đã phân tích
            </span>
          </div>

          {!studentReport ? (
            <EmptyState
              title="Đang tải báo cáo học sinh"
              text="Hệ thống đang tổng hợp dữ liệu video đã phân tích."
            />
          ) : studentReport.videos.length === 0 ? (
            <EmptyState
              title="Chưa có dữ liệu video cho học sinh này"
              text="Hãy phân tích video và gán Track cho đúng học sinh trong buổi học."
            />
          ) : (
            <>
              <div className="student-report-summary">
                <div>
                  <span>Video có dữ liệu</span>
                  <strong>{studentReport.summary.video_count}</strong>
                </div>
                <div>
                  <span>Bản ghi AI</span>
                  <strong>{studentReport.summary.ai_records}</strong>
                </div>
                <div>
                  <span>Tổng cảnh báo</span>
                  <strong>{studentReport.summary.total_alerts}</strong>
                </div>
                <div>
                  <span>MT / LA</span>
                  <strong>
                    {studentReport.summary.distracted}
                    {" / "}
                    {studentReport.summary.anxiety}
                  </strong>
                </div>
              </div>

              <div className="table-scroll student-video-report-table">
                <table>
                  <thead>
                    <tr>
                      <th>Video</th>
                      <th>Buổi học</th>
                      <th>Ngày phân tích</th>
                      <th>Bản ghi AI</th>
                      <th>Cảnh báo</th>
                      <th>Mất tập trung</th>
                      <th>Dấu hiệu lo âu</th>
                    </tr>
                  </thead>
                  <tbody>
                    {studentReport.videos.map((item) => (
                      <tr key={item.video_id}>
                        <td>
                          <strong>{item.video_name}</strong>
                        </td>
                        <td>{item.session_title || "-"}</td>
                        <td>
                          {item.analyzed_at
                            ? new Date(item.analyzed_at).toLocaleString("vi-VN")
                            : "-"}
                        </td>
                        <td>{item.ai_records}</td>
                        <td>{item.total_alerts}</td>
                        <td>{item.distracted}</td>
                        <td>{item.anxiety}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      )}

      <section className="dashboard-section">
        <div className="section-header">
          <div>
            <h2>Xu hướng cảnh báo</h2>
            <p>
              Số cảnh báo theo từng ngày trong khoảng đã chọn
            </p>
          </div>
        </div>

        {trends.length === 0 ? (
          <EmptyState
            title="Không có dữ liệu"
            text="Không tìm thấy dữ liệu thống kê trong khoảng thời gian đã chọn."
          />
        ) : (
          <div className="trend-list">
            {trends.map((item) => (
              <div
                className="trend-row"
                key={item.day}
              >
                <span>
                  {new Date(
                    item.day
                  ).toLocaleDateString(
                    "vi-VN",
                    {
                      day: "2-digit",
                      month: "2-digit",
                      year:
                        trends.length > 60
                          ? "2-digit"
                          : undefined,
                    }
                  )}
                </span>

                <div className="trend-bars">
                  <div
                    className="trend-bar total"
                    style={{
                      width: `${
                        (Number(item.total) / max) * 100
                      }%`,
                    }}
                  />
                </div>

                <strong>
                  {item.total}
                </strong>

                <small>
                  MT: {item.distracted} • LA: {item.anxiety}
                </small>
              </div>
            ))}
          </div>
        )}
      </section>
    </>
  );
}

function GradcamPage({
  examples,
}) {
  return (
    <>
      <PageHeader
        title="Grad-CAM"
        subtitle="Giải thích vùng ảnh ảnh hưởng đến dự đoán FER"
      />

      <section className="dashboard-section">
        <p className="info-note">
          Heatmap chỉ hỗ trợ giải thích
          vùng ảnh ảnh hưởng tới dự đoán
          của mô hình; không dùng để
          chẩn đoán tâm lý.
        </p>

        {examples.length === 0 ? (
          <EmptyState
            title="Chưa tìm thấy kết quả Grad-CAM"
            text="Đặt thư mục Week8_GradCAM_FER_V14 tại D:\Doan và khởi động lại Backend."
          />
        ) : (
          <div className="gradcam-grid">
            {examples.map(
              (item, index) => (
                <article
                  className="gradcam-card"
                  key={`${item.kind}-${item.true_label}-${index}`}
                >
                  <div className="gradcam-visual-pair">
                    <div className="gradcam-visual-item">
                      <span className="gradcam-visual-label">Ảnh gốc</span>
                      {item.original_url ? (
                        <ProtectedMedia path={item.original_url} alt={`Ảnh gốc - ${item.true_label}`} />
                      ) : (
                        <div className="image-placeholder">Không có ảnh gốc</div>
                      )}
                    </div>
                    <div className="gradcam-visual-item">
                      <span className="gradcam-visual-label">Heatmap Grad-CAM</span>
                      {item.heatmap_url ? (
                        <ProtectedMedia path={item.heatmap_url} alt={`Heatmap Grad-CAM - ${item.true_label}`} />
                      ) : (
                        <div className="image-placeholder">Không có heatmap</div>
                      )}
                    </div>
                  </div>

                  <div className="gradcam-info">
                    <span
                      className={
                        item.kind ===
                        "correct"
                          ? "badge normal"
                          : "badge distracted"
                      }
                    >
                      {item.kind ===
                      "correct"
                        ? "Đúng"
                        : "Sai"}
                    </span>

                    <strong>
                      {item.true_label} →{" "}
                      {item.pred_label}
                    </strong>

                    <span>
                      Confidence:{" "}
                      {Number(
                        item.confidence
                      ).toFixed(3)}
                    </span>
                  </div>
                </article>
              )
            )}
          </div>
        )}
      </section>
    </>
  );
}

function VideosPage({
  videos,
  classes,
  sessions,
  videoClassId,
  setVideoClassId,
  videoSessionId,
  setVideoSessionId,
  onCreateSession,
  isAdmin,
  fileInputRef,
  onUpload,
  onAnalyze,
  onDelete,
  onViewResults,
}) {
  const sessionOptions =
    sessions.filter(
      (item) =>
        String(item.class_id)
        === String(videoClassId)
    );

  return (
    <>
      <PageHeader
        title="Video lớp học"
        subtitle="Tải lên, phát và quản lý tiến trình phân tích AI"
        action={
          <div className="video-header-actions">
            <select
              value={videoClassId}
              onChange={(e) =>
                setVideoClassId(
                  e.target.value
                )
              }
            >
              <option value="">
                -- Chọn lớp --
              </option>
              {classes.map((item) => (
                <option
                  key={item.id}
                  value={item.id}
                >
                  {item.name}
                </option>
              ))}
            </select>

            <select
              value={videoSessionId}
              disabled={!videoClassId}
              onChange={(e) =>
                setVideoSessionId(
                  e.target.value
                )
              }
            >
              <option value="">
                -- Chọn buổi học --
              </option>
              {sessionOptions.map(
                (item) => (
                  <option
                    key={item.id}
                    value={item.id}
                  >
                    {item.title}
                    {" • "}
                    {String(
                      item.session_date
                    ).slice(0, 10)}
                  </option>
                )
              )}
            </select>

            <button
              className="secondary-button"
              disabled={!videoClassId}
              onClick={onCreateSession}
            >
              + Buổi học
            </button>

            <label
              className={`primary-button file-button ${
                !videoClassId || !videoSessionId
                  ? "disabled-like"
                  : ""
              }`}
              title={
                !videoClassId || !videoSessionId
                  ? "Chọn Lớp + Buổi học trước khi tải video"
                  : "Tải video cho buổi học đã chọn"
              }
            >
              + Tải video
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={onUpload}
                disabled={
                  !videoClassId
                  || !videoSessionId
                }
              />
            </label>
          </div>
        }
      />

      <section className="dashboard-section">
        {videos.length === 0 ? (
          <EmptyState
            title="Chưa có video"
            text="Tải video lớp học để quản lý và phân tích."
          />
        ) : (
          <div className="video-grid">
            {videos.map((item) => (
              <article
                className="video-card"
                key={item.id}
              >
                <div className="video-preview-grid">

                  <div className="video-preview-panel">
                    <div className="video-preview-title">
                      Video gốc
                    </div>

                    <ProtectedMedia path={item.file_url} kind="video" />
                  </div>

                  {item.analysis_url && (
                    <div className="video-preview-panel analyzed-panel">

                      <div className="video-preview-title analyzed-title">
                        Video AI đã phân tích
                      </div>

                      <ProtectedMedia path={item.analysis_url} kind="video" />

                      <div className="analysis-help">
                        Video này chứa Track ID, trạng thái hành vi,
                        kết quả Fusion và cảnh báo được vẽ trực tiếp
                        lên từng khung hình.
                      </div>

                    </div>
                  )}

                </div>

                <div className="video-info">
                  <strong>
                    {item.original_name}
                  </strong>

                  <span className="video-class-line">
                    Lớp: <b>{item.class_name || "Chưa gán lớp"}</b>
                  </span>

                  <span className="video-class-line">
                    Buổi học:{" "}
                    <b>
                      {item.session_title
                        || "Chưa gán buổi học"}
                    </b>
                    {item.session_date && (
                      <>
                        {" • "}
                        {new Date(
                          `${String(item.session_date).slice(0,10)}T00:00:00`
                        ).toLocaleDateString(
                          "vi-VN"
                        )}
                      </>
                    )}
                  </span>

                  <span>
                    #
                    {item.id} •{" "}
                    {(
                      Number(
                        item.file_size || 0
                      ) /
                      1024 /
                      1024
                    ).toFixed(1)}{" "}
                    MB
                  </span>

                  <span>
                    Trạng thái:{" "}
                    <b>
                      {item.status ===
                      "uploaded"
                        ? "Đã tải"
                        : item.status ===
                          "processing"
                        ? "Đang phân tích"
                        : item.status ===
                          "completed"
                        ? "Hoàn thành"
                        : "Thất bại"}
                    </b>
                  </span>

                  {item.status === "processing" && (
                    <div className="analysis-progress-wrap">
                      <div className="analysis-progress-head">
                        <span>Tiến trình AI</span>
                        <strong>
                          {Math.round(
                            Number(item.progress || 0)
                          )}%
                        </strong>
                      </div>

                      <div className="analysis-progress-track">
                        <div
                          className="analysis-progress-bar"
                          style={{
                            width: `${Math.max(
                              2,
                              Math.min(
                                100,
                                Number(item.progress || 0)
                              )
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}

                  {item.analysis_url && (
                    <span className="analysis-ready">
                      ✓ Đã tạo video minh chứng AI
                    </span>
                  )}

                  <span>
                    Kết quả AI:{" "}
                    {item.result_count} •
                    Cảnh báo:{" "}
                    {item.alert_count}
                  </span>

                  {item.error_message && (
                    <span className="video-error">
                      {item.error_message}
                    </span>
                  )}
                </div>

                <div className="video-actions">
                  <button
                    className="primary-button"
                    disabled={
                      item.status ===
                      "processing"
                    }
                    onClick={() =>
                      onAnalyze(item)
                    }
                  >
                    {item.status ===
                    "processing"
                      ? "Đang phân tích..."
                      : item.analysis_url
                      ? "Phân tích lại bằng V14"
                      : "Phân tích bằng V14"}
                  </button>

                  <button
                    className="secondary-button"
                    onClick={() =>
                      onViewResults(item)
                    }
                  >
                    Xem kết quả
                  </button>

                  {isAdmin && (
                    <button
                      className="danger-button"
                      disabled={
                        item.status ===
                        "processing"
                      }
                      onClick={() =>
                        onDelete(item)
                      }
                    >
                      Xóa
                    </button>
                  )}
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </>
  );
}

function AccountPage({
  user,
  currentPassword,
  setCurrentPassword,
  newPassword,
  setNewPassword,
  confirmPassword,
  setConfirmPassword,
  onSubmit,
  isAdmin,
  users,
  onCreateUser,
  onEditUser,
  onChangeRole,
  onDeleteUser,
}) {
  return (
    <>
      <PageHeader
        title="Tài khoản"
        subtitle="Thông tin đăng nhập và đổi mật khẩu"
      />

      <section className="account-grid">
        <div className="bottom-card">
          <h2>
            Thông tin tài khoản
          </h2>

          <div className="profile-list">
            <div>
              <span>Họ tên</span>
              <strong>
                {user?.name || "-"}
              </strong>
            </div>

            <div>
              <span>Email</span>
              <strong>
                {user?.email || "-"}
              </strong>
            </div>

            <div>
              <span>Quyền</span>
              <strong>
                {user?.role === "admin"
                  ? "Quản trị viên"
                  : "Giáo viên"}
              </strong>
            </div>
          </div>
        </div>

        <form
          className="bottom-card password-form"
          onSubmit={onSubmit}
        >
          <h2>Đổi mật khẩu</h2>

          <label>
            Mật khẩu hiện tại
          </label>
          <input
            type="password"
            value={currentPassword}
            onChange={(e) =>
              setCurrentPassword(
                e.target.value
              )
            }
            required
          />

          <label>
            Mật khẩu mới
          </label>
          <input
            type="password"
            value={newPassword}
            onChange={(e) =>
              setNewPassword(
                e.target.value
              )
            }
            minLength={8}
            required
          />

          <label>
            Xác nhận mật khẩu mới
          </label>
          <input
            type="password"
            value={confirmPassword}
            onChange={(e) =>
              setConfirmPassword(
                e.target.value
              )
            }
            minLength={8}
            required
          />

          <button
            className="primary-button"
            type="submit"
          >
            Đổi mật khẩu
          </button>
        </form>
      </section>

      {isAdmin && (
        <section className="dashboard-section">
          <div className="section-header section-header-row">
            <div>
              <h2>Phân quyền tài khoản</h2>
              <p>
                Admin có thể tạo, sửa, xóa tài khoản và thay đổi quyền.
              </p>
            </div>

            <button
              className="primary-button"
              onClick={onCreateUser}
            >
              + Tạo tài khoản
            </button>
          </div>

          <div className="table-wrap account-user-table">
            <table>
              <thead>
                <tr>
                  <th>Họ tên</th>
                  <th>Email</th>
                  <th>Quyền</th>
                  <th>Ngày tạo</th>
                  <th>Thao tác</th>
                </tr>
              </thead>
              <tbody>
                {users.map((item) => (
                  <tr key={item.id}>
                    <td>
                      <strong>
                        {item.full_name}
                      </strong>
                    </td>
                    <td>{item.email}</td>
                    <td>
                      {item.role === "admin"
                        ? "Admin"
                        : "Teacher"}
                    </td>
                    <td>
                      {new Date(
                        item.created_at
                      ).toLocaleDateString(
                        "vi-VN"
                      )}
                    </td>
                    <td>
                      <div className="row-actions account-actions">
                        <button
                          className="secondary-button"
                          onClick={() =>
                            onEditUser(item)
                          }
                        >
                          Sửa
                        </button>

                        <button
                          className="secondary-button"
                          onClick={() =>
                            onChangeRole(item)
                          }
                        >
                          Đổi quyền
                        </button>

                        <button
                          className="danger-button"
                          onClick={() =>
                            onDeleteUser(item)
                          }
                          disabled={
                            Number(item.id) ===
                            Number(user?.id)
                          }
                          title={
                            Number(item.id) ===
                            Number(user?.id)
                              ? "Không thể tự xóa tài khoản đang đăng nhập."
                              : "Xóa tài khoản"
                          }
                        >
                          Xóa
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}

function EmptyState({
  title,
  text,
}) {
  return (
    <div className="empty-state">
      <div className="empty-title">
        {title}
      </div>

      {text && (
        <div className="empty-description">
          {text}
        </div>
      )}
    </div>
  );
}

export default App;
