(function() {
    const baseFontSize = 9;

    const raw = window.__ERROR_CHART_PAYLOAD__;
    const payload = typeof raw === "string" ? JSON.parse(raw) : raw;

    const el = document.getElementById("chart");
    if (!el) return;

    if (!payload || !payload.dates || !payload.bias) {
        el.innerHTML =
            "<div style='padding:12px;color:#64748b'>No bias data available.</div>";
        return;
    }

    if (typeof echarts === "undefined") {
        el.innerHTML =
            "<div style='padding:12px;color:#64748b'>ECharts not loaded.</div>";
        return;
    }

    const dates = payload.dates;
    const bias = payload.bias;

    const colors = bias.map(v =>
        v >= 0 ? "#ef4444" : "#10b981"
    );

    const maxAbs = Math.max(...bias.map(v => Math.abs(v)));
    const yMax = Math.ceil(maxAbs / 5) * 5 || 5;

    const chart = echarts.init(el);

    chart.setOption({
        tooltip: {
            trigger: "axis",
            formatter: (p) => {
                const v = p[0].data;
                const dir = v > 0 ? "Overestimate" : "Underestimate";
                return `
          <b>${p[0].axisValue}</b><br/>
          Bias: <b>${v.toFixed(2)} mm</b><br/>
          <span style="color:#64748b">${dir}</span>
        `;
            }
        },

        grid: {
            left: 36,
            right: 12,
            top: 16,
            bottom: 32
        },

        xAxis: {
            type: "category",
            data: dates,
            axisLabel: {
                color: "#64748b",
                fontSize: baseFontSize,
                interval: "auto"
            },
            axisLine: {
                lineStyle: {
                    color: "#cbd5f5"
                }
            }
        },

        yAxis: {
            type: "value",
            min: -yMax,
            max: yMax,
            name: "Bias (mm)",
            nameTextStyle: {
                color: "#64748b",
                fontSize: baseFontSize - 1
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

        series: [{
            type: "bar",
            data: bias,
            barWidth: "90%",
            itemStyle: {
                color: (p) => colors[p.dataIndex]
            },

            markLine: {
                silent: true,
                symbol: "none",
                lineStyle: {
                    color: "#94a3b8",
                    type: "dashed",
                    width: 1
                },
                data: [{
                    yAxis: 0
                }]
            }
        }]
    });

    window.addEventListener("resize", () => chart.resize());
})();