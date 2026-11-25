(() => {
  "use strict";
  const scales = document.querySelectorAll("[data-freq-select]");
  const setScale = (value) => {
    document.documentElement.dataset.freqScale = value;
    scales.forEach((select) => { select.value = value; });
  };
  scales.forEach((select) => {
    select.addEventListener("change", () => setScale(select.value));
    select.closest("label").hidden = false;
  });
  if (scales.length) setScale(scales[0].value);

  const all = [...document.querySelectorAll("audio")];
  // Only one player sounds at a time across the page.
  document.addEventListener("play", (event) => {
    if (!(event.target instanceof HTMLAudioElement)) return;
    all.forEach((p) => { if (p !== event.target) p.pause(); });
  }, true);
  document.addEventListener("shown.bs.tab", () => {
    all.forEach((p) => {
      const pane = p.closest(".tab-pane");
      if (pane && !pane.classList.contains("active")) p.pause();
    });
  });

  document.querySelectorAll("[data-ab]").forEach((root) => {
    const tracks = {};
    root.querySelectorAll("[data-ab-track]").forEach((fig) => {
      tracks[fig.dataset.abTrack] = { fig, audio: fig.querySelector("audio") };
    });
    const play = root.querySelector("[data-ab-play]");
    const loop = root.querySelector("[data-ab-loop]");
    const time = root.querySelector("[data-ab-time]");
    const picks = [...root.querySelectorAll("[data-ab-pick]")];
    let active = "a";

    const playing = () => Object.values(tracks).some((t) => !t.audio.paused);
    const render = () => {
      picks.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.abPick === active)));
      Object.entries(tracks).forEach(([k, t]) => t.fig.classList.toggle("is-active", k === active));
      const on = playing();
      play.textContent = on ? "❚❚ Pause" : "▶ Play";
      play.setAttribute("aria-pressed", String(on));
      time.textContent = `${tracks[active].audio.currentTime.toFixed(2)} s`;
    };
    // Switch source at the current playback position; keep playing if it was.
    const select = (key) => {
      if (key === active) return;
      const from = tracks[active].audio, to = tracks[key].audio;
      const wasPlaying = !from.paused, at = from.currentTime;
      from.pause();
      to.currentTime = Math.min(at, Number.isFinite(to.duration) ? to.duration : at);
      active = key;
      if (wasPlaying) to.play();
      render();
    };

    picks.forEach((b) => b.addEventListener("click", () => select(b.dataset.abPick)));
    play.addEventListener("click", () => {
      const a = tracks[active].audio;
      if (a.paused) a.play(); else a.pause();
    });
    const applyLoop = () => Object.values(tracks).forEach((t) => { t.audio.loop = loop.checked; });
    loop.addEventListener("change", applyLoop);
    Object.entries(tracks).forEach(([k, t]) => {
      // Using a track's own native controls makes it the active side.
      t.audio.addEventListener("play", () => {
        if (k !== active) {
          const other = tracks[active].audio;
          other.pause();
          active = k;
        }
        render();
      });
      ["pause", "ended", "timeupdate", "seeked"].forEach((e) => t.audio.addEventListener(e, render));
    });
    root.addEventListener("keydown", (event) => {
      if (event.target instanceof HTMLInputElement) return;
      if (event.key === "a" || event.key === "A") select("a");
      if (event.key === "b" || event.key === "B") select("b");
    });
    applyLoop();
    root.querySelector(".ab-controls").hidden = false;
    render();
  });
})();
