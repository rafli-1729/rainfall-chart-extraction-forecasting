document.querySelectorAll('.kpi-value[data-target]').forEach(el => {
    const raw = el.dataset.target;
    const target = parseFloat(raw);

    if (isNaN(target)) {
        return;
    }

    let current = 0;
    const direction = target >= 0 ? 1 : -1;
    const absTarget = Math.abs(target);
    const step = Math.max(0.01, absTarget / 40);

    function tick() {
        current += step;

        if (current >= absTarget) {
            el.textContent = (direction * absTarget)
                .toFixed(2)
                .replace(/\.00$/, "");
        } else {
            el.textContent = (direction * current).toFixed(1);
            requestAnimationFrame(tick);
        }
    }

    tick();
});
