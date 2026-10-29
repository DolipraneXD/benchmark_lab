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
    // benchmark hover cards: open on hover or keyboard focus, stay open while the pointer
    // is on the card (so its link can be clicked), placed in the viewport (position: fixed)
    let openTip = null, closeTimer = null;
    const place = (el) => {
      const tip = el.querySelector(".tip");
      const r = el.getBoundingClientRect();
      const w = tip.offsetWidth, h = tip.offsetHeight;
      const left = Math.max(16, Math.min(r.left, window.innerWidth - w - 16));
      const below = r.bottom + 6 + h < window.innerHeight;
      tip.style.left = left + "px";
      tip.style.top = (below ? r.bottom + 6 : Math.max(8, r.top - h - 6)) + "px";
    };
    const show = (el) => {
      clearTimeout(closeTimer);
      if (openTip && openTip !== el) openTip.classList.remove("open");
      openTip = el;
      place(el);
      el.classList.add("open");
      place(el); // again once visible: sizes are exact now
    };
    const hideSoon = () => {
      clearTimeout(closeTimer);
      closeTimer = setTimeout(() => {
        if (openTip) openTip.classList.remove("open");
        openTip = null;
      }, 200);
    };
    document.addEventListener("mouseover", (e) => {
      const el = e.target.closest(".bench-tip");
      if (el) show(el);
      else if (openTip) hideSoon();
    });
    document.addEventListener("focusin", (e) => {
      const el = e.target.closest(".bench-tip");
      if (el) show(el);
      else if (openTip) hideSoon();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && openTip) { openTip.classList.remove("open"); openTip = null; }
    });
    window.addEventListener("scroll", () => openTip && place(openTip), { passive: true });
    // a click on the card's link must not toggle the checkbox of the label around it
    document.addEventListener("click", (e) => {
      if (e.target.closest(".bench-tip .tip") && !e.target.closest("a")) e.preventDefault();
    });

    // model form: show the fields the chosen provider needs, with its own hints
    const prov = document.querySelector("[data-provider]");
    if (prov) {
      const form = prov.closest("form");
      const field = (n) => form.querySelector(`[data-field='${n}']`);
      const sync = () => {
        const o = prov.selectedOptions[0].dataset;
        field("base").hidden = o.base === "none";
        field("version").hidden = !o.version;
        field("region").hidden = !o.region;
        form.querySelector("#f-base").required = o.base === "required";
        form.querySelector("[data-base-hint]").textContent =
          o.base === "required" ? "required" : o.baseDefault ? `default ${o.baseDefault}` : "optional";
        form.querySelector("#f-model").placeholder = o.example ? `e.g. ${o.example}` : "";
        form.querySelector("[data-provider-note]").textContent = o.key
          ? "Needs an API key (below)."
          : "Usually no API key: choose \u201cNo key needed\u201d below unless yours has one.";
      };
      prov.addEventListener("change", sync);
      sync();
    }

    // test list on a run page: paging, and back to page 1 when a filter changes
    document.addEventListener("click", (e) => {
      const b = e.target.closest("[data-page]");
      const f = document.getElementById("tests-filter");
      if (!b || !f) return;
      f.elements.page.value = b.dataset.page;
      f.dispatchEvent(new Event("change"));
      document.getElementById("tests").scrollIntoView({ block: "start", behavior: "smooth" });
    });
    const tf = document.getElementById("tests-filter");
    if (tf) {
      tf.addEventListener("input", (e) => {
        if (e.target.name === "page") return;
        tf.elements.page.value = 1;
        if (e.target.type === "search") tf.dispatchEvent(new Event("change")); // live search
      }, true);
      tf.addEventListener("submit", (e) => e.preventDefault());
    }

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
