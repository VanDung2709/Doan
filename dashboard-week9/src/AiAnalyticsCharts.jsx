import React, { useMemo, useState } from "react";
import "./AiAnalyticsCharts.css";

// WEEK9_AI_ANALYTICS_V1 — Only summarized alert counts from authorized PostgreSQL APIs.
// No browser-side guesses about student identity, emotions or mental health.
const NUMBER_FORMAT = new Intl.NumberFormat("vi-VN");

function toDayLabel(raw) {
  const datePart = String(raw ?? "").slice(0, 10);
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(datePart);
  return match ? `${match[3]}/${match[2]}` : datePart;
}

function dailyPoints(trends) {
  const rows = (Array.isArray(trends) ? trends : [])
    .filter((row) => /^\d{4}-\d{2}-\d{2}/.test(String(row?.day ?? "")))
    .map((row) => ({
      day: String(row.day).slice(0, 10),
      value: Math.max(0, Number(row.total) || 0),
    }))
    .sort((a, b) => a.day.localeCompare(b.day));
  // Avoid rendering thousands of interactive dots for a 10-year selected range.
  const stride = Math.max(1, Math.ceil(rows.length / 115));
  if (stride === 1) return rows.map((row) => ({ ...row, label: toDayLabel(row.day) }));
  const grouped = [];
  for (let i = 0; i < rows.length; i += stride) {
    const slice = rows.slice(i, i + stride);
    grouped.push({
      day: slice[0].day,
      label: `${toDayLabel(slice[0].day)}–${toDayLabel(slice[slice.length - 1].day)}`,
      value: slice.reduce((sum, row) => sum + row.value, 0),
    });
  }
  return grouped;
}

function HourlyChartData(hourly) {
  const incoming = Array.isArray(hourly?.hours) ? hourly.hours : [];
  const byHour = new Map(incoming.map((row) => [Number(row.hour), Math.max(0, Number(row.total) || 0)]));
  return Array.from({ length: 24 }, (_, hour) => ({
    day: String(hour),
    label: `${String(hour).padStart(2, "0")}:00`,
    value: byHour.get(hour) || 0,
  }));
}

function smoothPath(points) {
  if (!points.length) return "";
  if (points.length === 1) return `M ${points[0].x} ${points[0].y}`;
  let path = `M ${points[0].x} ${points[0].y}`;
  for (let i = 0; i < points.length - 1; i += 1) {
    const a = points[i], b = points[i + 1];
    const midX = (a.x + b.x) / 2;
    path += ` C ${midX} ${a.y}, ${midX} ${b.y}, ${b.x} ${b.y}`;
  }
  return path;
}

function TrendGraph({ rows, hourly = false }) {
  const [active, setActive] = useState(null);
  const data = Array.isArray(rows) ? rows : [];
  const width = 960, height = 310;
  const left = 54, right = 18, top = 31, bottom = 258;
  const availableWidth = width - left - right;
  const availableHeight = bottom - top;
  const peak = Math.max(1, ...data.map((row) => row.value));
  const ceiling = peak <= 4 ? 4 : Math.ceil(peak / 4) * 4;
  const points = data.map((row, i) => ({
    ...row,
    x: left + availableWidth * (data.length <= 1 ? 0.5 : i / (data.length - 1)),
    y: bottom - (row.value / ceiling) * availableHeight,
  }));
  const curve = smoothPath(points);
  const fillPath = points.length > 1
    ? `${curve} L ${points[points.length - 1].x} ${bottom} L ${points[0].x} ${bottom} Z`
    : "";
  const hovered = active === null ? null : points[active];
  const axisStep = hourly ? 3 : Math.max(1, Math.ceil((points.length - 1) / 9));
  const axes = points.filter((_, i) => i === 0 || i === points.length - 1 || i % axisStep === 0);
  const labels = new Set();
  const showData = data.some((row) => row.value > 0);

  return (
    <div className="ai-vis-chart-wrap">
      {(!data.length || !showData) && (
        <div className="ai-vis-empty" role="status">
          {hourly ? "Chưa ghi nhận cảnh báo hôm nay." : "Không có cảnh báo trong khoảng thời gian đã chọn."}
        </div>
      )}
      <svg
        className="ai-vis-line-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={hourly ? "Số cảnh báo theo từng giờ hôm nay" : "Xu hướng số cảnh báo theo ngày"}
        onPointerLeave={() => setActive(null)}
      >
        <defs>
          <linearGradient id={hourly ? "ai-vis-today-fill" : "ai-vis-trend-fill"} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#686cf6" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#686cf6" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 1, 2, 3, 4].map((index) => {
          const y = bottom - (index / 4) * availableHeight;
          return (
            <g key={index}>
              <line x1={left} x2={width - right} y1={y} y2={y} stroke="#29374b" strokeDasharray="3 6" />
              <text className="ai-vis-tick" x={left - 9} y={y + 4} textAnchor="end">
                {NUMBER_FORMAT.format((ceiling / 4) * index)}
              </text>
            </g>
          );
        })}
        {axes.map((point) => {
          if (labels.has(point.x)) return null;
          labels.add(point.x);
          return <text key={`${point.day}-${point.x}`} className="ai-vis-tick" x={point.x} y={bottom + 23} textAnchor="middle">{point.label}</text>;
        })}
        {showData && points.length > 1 && (
          <>
            <path d={fillPath} fill={`url(#${hourly ? "ai-vis-today-fill" : "ai-vis-trend-fill"})`} />
            <path d={curve} fill="none" stroke="#686cf6" strokeWidth="3.2" strokeLinecap="round" />
          </>
        )}
        {hovered && showData && (
          <g pointerEvents="none">
            <line x1={hovered.x} x2={hovered.x} y1={top} y2={bottom} stroke="#92a4c8" strokeDasharray="4 4" />
            <circle cx={hovered.x} cy={hovered.y} r="6" fill="#686cf6" stroke="#fff" strokeWidth="2" />
            <rect x={Math.max(70, Math.min(hovered.x - 81, width - 174))} y={Math.max(4, hovered.y - 78)} width="162" height="58" rx="9" fill="#edf2ff" />
            <text x={Math.max(70, Math.min(hovered.x - 81, width - 174)) + 12} y={Math.max(4, hovered.y - 78) + 22} fill="#111827" fontSize="13" fontWeight="600">{hovered.label}</text>
            <text x={Math.max(70, Math.min(hovered.x - 81, width - 174)) + 12} y={Math.max(4, hovered.y - 78) + 42} fill="#434fce" fontSize="13">{NUMBER_FORMAT.format(hovered.value)} cảnh báo</text>
          </g>
        )}
        {/* Accessible hover region, scaled to the responsive SVG viewBox. */}
        {points.length > 0 && <rect
          x={left} y={top} width={availableWidth} height={availableHeight}
          fill="transparent"
          onPointerMove={(event) => {
            const bounds = event.currentTarget.ownerSVGElement?.getBoundingClientRect();
            if (!bounds?.width) return;
            const svgX = ((event.clientX - bounds.left) / bounds.width) * width;
            const index = Math.max(0, Math.min(points.length - 1, Math.round((svgX - left) / availableWidth * (points.length - 1))));
            setActive(index);
          }}
        />}
      </svg>
    </div>
  );
}

function DashboardCard({ title, subtitle, badge, children }) {
  return (
    <section className="ai-vis-card">
      <div className="ai-vis-card-head">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        {badge != null && <span className="ai-vis-badge">{badge}</span>}
      </div>
      <div className="ai-vis-card-body">{children}</div>
    </section>
  );
}

function AlertDonut({ trends }) {
  const totals = (Array.isArray(trends) ? trends : []).reduce((sum, row) => ({
    all: sum.all + Math.max(0, Number(row.total) || 0),
    distracted: sum.distracted + Math.max(0, Number(row.distracted) || 0),
    anxiety: sum.anxiety + Math.max(0, Number(row.anxiety) || 0),
  }), { all: 0, distracted: 0, anxiety: 0 });
  const segments = [
    { label: "Mất tập trung", count: totals.distracted, color: "#f59e0b" },
    { label: "Dấu hiệu lo âu", count: totals.anxiety, color: "#a78bfa" },
    { label: "Khác", count: Math.max(0, totals.all - totals.distracted - totals.anxiety), color: "#38bdf8" },
  ].filter((item) => item.count > 0);
  const sum = segments.reduce((acc, item) => acc + item.count, 0);
  const circumference = 2 * Math.PI * 84;
  let progress = 0;

  return (
    <div className="ai-vis-donut-layout">
      <div className="ai-vis-donut-wrap">
        <svg viewBox="0 0 290 290" role="img" aria-label="Biểu đồ phân bố loại cảnh báo">
          <circle cx="145" cy="145" r="84" stroke="#25344c" strokeWidth="34" fill="none" />
          {segments.map((segment) => {
            const fraction = sum ? segment.count / sum : 0;
            const offset = progress;
            progress += fraction;
            return <circle
              key={segment.label}
              cx="145" cy="145" r="84" fill="none" stroke={segment.color} strokeWidth="34"
              strokeDasharray={`${Math.max(0, fraction * circumference - 2)} ${circumference}`}
              strokeDashoffset={-offset * circumference}
              transform="rotate(-90 145 145)"
            />;
          })}
          <text x="145" y="141" textAnchor="middle" className="ai-vis-total">{NUMBER_FORMAT.format(totals.all)}</text>
          <text x="145" y="162" textAnchor="middle" className="ai-vis-total-sub">cảnh báo</text>
        </svg>
      </div>
      <div className="ai-vis-donut-legend">
        {(segments.length ? segments : [{ label: "Chưa có cảnh báo", count: 0, color: "#64748b" }]).map((item) => (
          <div className="ai-vis-legend-line" key={item.label}>
            <div className="ai-vis-legend-name"><span className="ai-vis-legend-dot" style={{ background: item.color }} />{item.label}</div>
            <strong>{NUMBER_FORMAT.format(item.count)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export function StatsAnalyticsCharts({ trends = [] }) {
  const rows = useMemo(() => dailyPoints(trends), [trends]);
  return (
    <div className="ai-vis-stack" data-ai-analytics="stats">
      <DashboardCard title="Biểu đồ xu hướng cảnh báo" subtitle="Số cảnh báo theo thời gian và bộ lọc đã chọn từ PostgreSQL">
        <TrendGraph rows={rows} />
      </DashboardCard>
      <DashboardCard title="Phân bố loại cảnh báo" subtitle="Cơ cấu số cảnh báo trong khoảng thời gian đang xem">
        <AlertDonut trends={trends} />
      </DashboardCard>
    </div>
  );
}

export function TodayOverviewChart({ hourly, error }) {
  const rows = useMemo(() => HourlyChartData(hourly), [hourly]);
  const total = rows.reduce((sum, row) => sum + row.value, 0);
  return (
    <div className="ai-vis-stack" data-ai-analytics="today">
      <DashboardCard title="Diễn biến cảnh báo hôm nay" subtitle="Số cảnh báo theo giờ tại Việt Nam (UTC+7)" badge={hourly ? `${NUMBER_FORMAT.format(total)} cảnh báo` : "Đang tải"}>
        {error ? <p className="ai-vis-error" role="status">Chưa tải được biểu đồ từ PostgreSQL: {error}</p> :
          hourly ? <TrendGraph hourly rows={rows} /> : <div className="ai-vis-loading" role="status">Đang tải số liệu hôm nay từ PostgreSQL…</div>}
      </DashboardCard>
    </div>
  );
}
