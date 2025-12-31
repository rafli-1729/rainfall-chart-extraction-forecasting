(function() {
    const baseFontSize = 9;

    const payload = typeof window.__ERROR_DIST_PAYLOAD__ === "string" ?
        JSON.parse(window.__ERROR_DIST_PAYLOAD__) :
        window.__ERROR_DIST_PAYLOAD__;

    const el = document.getElementById("chart");
    if (!el) return;

    if (!payload || !payload.values || payload.values.length === 0) {
        el.innerHTML =
            "<div style='padding:12px;color:#64748b'>No error distribution data.</div>";
        return;
    }

    if (typeof echarts === "undefined") {
        el.innerHTML =
            "<div style='padding:12px;color:#64748b'>ECharts not loaded.</div>";
        return;
    }

    const values = payload.values;

    const binSize = 1; // 0.25 mm
    const maxVal = Math.max(...values);
    const binCount = Math.ceil(maxVal / binSize);

    const bins = Array(binCount).fill(0);
    values.forEach(v => {
        const idx = Math.min(
            Math.floor(v / binSize),
            binCount - 1
        );
        bins[idx] += 1;
    });

    const labels = bins.map((_, i) => {
        const left  = i * binSize;
        const right = (i + 1) * binSize;
        return `${left.toFixed(1)}–${right.toFixed(1)}`;
    });

    const chart = echarts.init(el);

    chart.setOption({
        tooltip: {
            trigger: "axis",
            axisPointer: {
                type: "shadow"
            },
            formatter: (p) =>
                `<b>${p[0].axisValue} mm</b><br/>Count: <b>${p[0].data}</b>`
        },

        grid: {
            left: 32,
            right: 12,
            top: 16,
            bottom: 32
        },

        xAxis: {
            type: "category",
            data: labels,
            axisLabel: {
                color: "#64748b",
                fontSize: baseFontSize
            },
            axisLine: {
                lineStyle: {
                    color: "#cbd5f5"
                }
            }
        },

        yAxis: {
            type: "value",
            name: "Days",
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
            data: bins,
            barWidth: "90%",
            itemStyle: {
                color: "#ef4444"
            }
        }]
    });

    window.addEventListener("resize", () => chart.resize());
})();