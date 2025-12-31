(function () {
  const baseFontSize = 9;

  const raw = window.__CALIBRATION_PAYLOAD__ ;
  const payload = typeof raw === "string" ? JSON.parse(raw) : raw;

  const el = document.getElementById("chart");
  if (!el) return;

  if (!payload || !payload.x || !payload.y || payload.x.length === 0) {
    el.innerHTML =
      "<div style='padding:12px;color:#64748b'>No calibration data available.</div>";
    return;
  }

  if (typeof echarts === "undefined") {
    el.innerHTML =
      "<div style='padding:12px;color:#64748b'>ECharts not loaded.</div>";
    return;
  }

  const points = payload.x.map((x, i) => [x, payload.y[i]]);
  const rawMax = Math.max(
    Math.max(...payload.x),
    Math.max(...payload.y)
  );

  const maxVal = Math.ceil(rawMax / 10) * 10;

  const chart = echarts.init(el);

  chart.setOption({
    tooltip: {
      trigger: "item",
      formatter: (p) =>
        `Extracted: <b>${p.data[0].toFixed(1)}</b> mm<br/>
         Predicted: <b>${p.data[1].toFixed(1)}</b> mm`
    },

    grid: {
      left: 0,
      right: 0,
      top: 16,
      bottom: 32
    },

    xAxis: {
        type: "value",
        min: 0,
        max: maxVal,

        name: "Extracted (mm)",
        nameLocation: "middle",
        nameGap: 28,
        nameTextStyle: {
            color: "#64748b",
            fontSize: baseFontSize
        },

        axisLabel: {
            color: "#64748b",
            fontSize: baseFontSize
        },

        splitLine: {
            lineStyle: {
            color: "#e5e7eb",
            type: "dashed"
            }
        }
    },

    yAxis: {
      type: "value",
      name: "Predicted (mm)",
      fontsize: baseFontSize,
      nameTextStyle: {
        color: "#64748b",
        fontSize: baseFontSize - 1,
        padding: [0, 0, 0, 0]
    },
      min: 0,
      max: maxVal,
      axisLabel: {
        color: "#64748b",
        fontSize: baseFontSize
      },
      splitLine: {
        lineStyle: { color: "#e5e7eb", type: "dashed" }
      }
    },

    series: [
      {
        name: "Prediction",
        type: "scatter",
        data: points,
        symbolSize: 4,
        itemStyle: {
          color: "rgba(59, 130, 246, 0.35)"
        }
      },
      {
        name: "Ideal",
        type: "line",
        data: [[0, 0], [maxVal, maxVal]],
        symbol: "none",
        lineStyle: {
          color: "#94a3b8",
          type: "dashed",
          width: 1
        }
      }
    ]
  });

  setTimeout(() => chart.resize(), 0);
  window.addEventListener("resize", () => chart.resize());
})();
