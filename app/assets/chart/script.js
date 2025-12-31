(function () {
  const baseFontSize = 9;

  const payload = window.__LINE_CHART_PAYLOAD__;
  if (!payload || !payload.dates || !payload.series) {
    const el = document.getElementById("chart");
    if (el) el.innerHTML = "<div style='padding:12px;color:#64748b'>No data payload provided.</div>";
    return;
  }

  const el = document.getElementById("chart");
  if (!el) return;

  if (typeof echarts === "undefined") {
    el.innerHTML = "<div style='padding:12px;color:#64748b'>ECharts not loaded.</div>";
    return;
  }

  const chart = echarts.init(el);

  const colors = {
    Observed: "#0f172a",
    Predicted: "#3b82f6",
    Extracted: "#10b981"
  };

  const seriesOrder = ["Observed", "Predicted", "Extracted"];

  const series = seriesOrder
    .filter(name => payload.series[name])
    .map(name => ({
      name,
      type: "line",
      data: payload.series[name],
      smooth: true,
      symbol: "circle",
      symbolSize: 2.5,
      lineStyle: {
        width: 1.5,
        type: name === "Predicted" ? "dashed" : "solid"
      },
      color: colors[name] || undefined
    }));

  const option = {
    tooltip: {
      trigger: "axis",
      backgroundColor: "#ffffff",
      borderColor: "#e5e7eb",
      borderWidth: 1,
      textStyle: {
        color: "#0f172a",
        fontSize: baseFontSize
     },
      formatter: (params) => {
        const date = params[0].axisValue;
        return [
            `<b>${date}</b>`,
            ...params.map(p => {
            const val = p.data != null ? p.data.toFixed(1) : "NA";
            return `${p.marker} ${p.seriesName}: <b>${val}</b> mm`;
            })
        ].join("<br/>");
      }

    },
    legend: {
      bottom: 0,
      itemWidth: 14,
      itemHeight: 6,
      itemGap: 12,
      textStyle: {
        color: "#475569",
        fontSize: baseFontSize
     }
    },
    grid: {
      left: 44,
      right: 20,
      top: 18,
      bottom: 52
    },
    xAxis: {
      type: "category",
      data: payload.dates,
      axisLine: { lineStyle: { color: "#cbd5f5" } },
      axisLabel: {
        color: "#64748b",
        fontSize: baseFontSize
     }
    },
    yAxis: {
      type: "value",
      name: "mm",
      nameTextStyle: {
        color: "#4748b",
        fontSize: baseFontSize
     },
      axisLabel: {
        color: "#64748b",
        fontSize: baseFontSize
     },
      splitLine: { lineStyle: { color: "#e5e7eb", type: "dashed" } }
},
    series
  };

  chart.setOption(option);

  window.addEventListener("resize", () => chart.resize());
})();