(() => {
  "use strict";
  const root = document.getElementById("listening-comparison");
  const caseChoice = root.querySelector("#listening-case");
  const methodChoice = root.querySelector("#listening-method");
  const players = [...root.querySelectorAll("audio")];
  const refresh = () => {
    players.forEach((player) => player.pause());
    root.querySelectorAll(".listening-case").forEach((section) => {
      section.hidden = section.dataset.case !== caseChoice.value;
    });
    root.querySelectorAll(".listening-card.generated").forEach((card) => {
      card.hidden = card.dataset.method !== methodChoice.value;
    });
  };
  root.addEventListener("play", (event) => {
    if (event.target instanceof HTMLAudioElement) {
      players.forEach((player) => {
        if (player !== event.target) player.pause();
      });
    }
  }, true);
  caseChoice.addEventListener("change", refresh);
  methodChoice.addEventListener("change", refresh);
  root.querySelector(".listening-controls").hidden = false;
  refresh();
})();
