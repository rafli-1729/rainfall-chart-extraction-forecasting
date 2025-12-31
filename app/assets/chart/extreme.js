(function() {
    const baseFontSize = 9;

    const payload = typeof window.__ERROR_CHART_PAYLOAD__ === "string" ?
        JSON.parse(window.__ERROR_CHART_PAYLOAD__) :
        window.__ERROR_CHART_PAYLOAD__;

    if (!payload || !payload.dates || !payload.series) {
        const el = document.getElementById("chart");
        if (el) {
            el.innerHTML =
                "<div style='padding:12px;color:#64748b'>No error data available.</div>";
        }
        return;
    }

    const el = document.getElementById("chart");
    if (!el) return;

    if (typeof echarts === "undefined") {
        el.innerHTML =
            "<div style='padding:12px;color:#64748b'>ECharts not loaded.</div>";
        return;
    }

    const chart = echarts.init(el);
    const seriesData = payload.series["Extreme Error"] || [];

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
                const p = params[0];
                const val = p.data != null ? p.data.toFixed(1) : "NA";
                return `<b>${p.axisValue}</b><br/>
                ${p.marker} Absolute Error: <b>${val}</b> mm`;
            }
        },

        grid: {
            left: 0,
            right: 0,
            top: 18,
            bottom: 32
        },

        xAxis: {
            type: "category",
            data: payload.dates,
            boundaryGap: true,
            axisLine: {
                lineStyle: {
                    color: "#cbd5f5"
                }
            },
            axisLabel: {
                color: "#64748b",
                fontSize: baseFontSize,
                interval: "auto",
                formatter: (value) => {
                    const d = new Date(value);
                    return d.toLocaleDateString("en-US", {
                        month: "short",
                        day: "2-digit"
                    });
                }
            }
        },


        yAxis: {
            type: "value",
            name: "mm",
            nameTextStyle: {
                color: "#475569",
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

        series: [{
            name: "Absolute Error",
            type: "line",
            data: seriesData,
            smooth: true,
            symbol: "none",
            lineStyle: {
                width: 1.5,
                color: "#ef4444"
            },
            areaStyle: {
                color: "rgba(239, 68, 68, 0.12)"
            }
        }]
    };

    chart.setOption(option);
    window.addEventListener("resize", () => chart.resize());
})();