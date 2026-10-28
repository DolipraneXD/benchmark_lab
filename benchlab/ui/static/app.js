// BenchLab panel: live fragments without a framework.
//   <div data-poll="/fragments/x" data-every="2000">  -> refreshed in place
//   <form data-target="#id">                          -> POSTed with fetch, response into #id
//   <form data-target="#id" data-live>                -> also re-posted on every change
//   <button data-post="/url" data-target="#id">       -> POST, response into #id
//   <form data-confirm="Sure?">                       -> asks first
(function () {
  async function fill(el, url, opts) {
    try {
      const r = await fetch(url, Object.assign({ credentials: "same-origin" }, opts || {}));
      const html = await r.text();
      const log = el.querySelector(".log");
      const stick = !log || log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
      el.innerHTML = html;
      const nlog = el.querySelector(".log");
      if (nlog && stick) nlog.scrollTop = nlog.scrollHeight;
    } catch (e) {
      el.insertAdjacentHTML("afterbegin", "<p class='err'>connection lost; retrying…</p>");
    }
  }

  function poll(el) {
    const every = parseInt(el.dataset.every || "3000", 10);
    const tick = async () => {
      if (!document.hidden) await fill(el, el.dataset.poll);
      setTimeout(tick, every);
    };
    tick();
  }

  function submitTo(form) {
    const target = document.querySelector(form.dataset.target);
    target.classList.add("busy");
    fill(target, form.action, { method: "POST", body: new FormData(form) }).then(() =>
      target.classList.remove("busy")
    );
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-poll]").forEach(poll);
    document.querySelectorAll("form[data-target]").forEach((f) => {
      f.addEventListener("submit", (e) => {
        if (e.submitter && e.submitter.formAction && e.submitter.hasAttribute("formaction")) return;
        e.preventDefault();
        submitTo(f);
      });
      if (f.hasAttribute("data-live")) {
        let t;
        f.addEventListener("change", () => {
          clearTimeout(t);
          t = setTimeout(() => submitTo(f), 250);
        });
      }
    });
    document.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-post]");
      if (!b) return;
      e.preventDefault();
      const target = document.querySelector(b.dataset.target);
      b.disabled = true;
      target.innerHTML = "<span class='muted'>testing…</span>";
      fill(target, b.dataset.post, { method: "POST" }).then(() => (b.disabled = false));
    });
    document.querySelectorAll("form[data-confirm]").forEach((f) =>
      f.addEventListener("submit", (e) => {
        if (!confirm(f.dataset.confirm)) e.preventDefault();
      })
    );
    // quick-pick buttons on the run page: select benches of a stage / all models
    document.querySelectorAll("[data-pick]").forEach((b) =>
      b.addEventListener("click", () => {
        const [name, values] = b.dataset.pick.split(":");
        const want = values === "*" ? null : values.split(",");
        document.querySelectorAll(`input[name='${name}']`).forEach((i) => {
          i.checked = want === null ? true : want.includes(i.value);
        });
        const f = b.closest("form");
        if (f) f.dispatchEvent(new Event("change"));
      })
    );
  });
})();
