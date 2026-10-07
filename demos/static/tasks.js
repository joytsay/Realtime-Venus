import { t } from "./i18n.js?v=20261007-session-init";
const terminal = new Set(["delivered", "failed", "cancelled"]);
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("icon");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", "#i-" + name);
  svg.append(use);
  return svg;
}
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
export class TaskTray {
  constructor({ send, download }) {
    this.send = send;
    this.download = download;
    this.items = new Map();
    this.nodes = new Map();
    this.tray = document.getElementById("task-tray");
    this.list = document.getElementById("task-list");
    this.toggle = document.getElementById("tasks-toggle");
    this.active = false;
    this.open = false;
    this.toggle.onclick = () => this.show(!this.open);
    document.getElementById("close-tasks").onclick = () => this.show(false);
    window.addEventListener("venus-language", () => this.render());
    setInterval(() => this.updateTime(), 1000);
  }
  show(value) {
    this.open = value;
    this.tray.classList.toggle("open", value);
    this.tray.inert = !value;
    this.tray.setAttribute("aria-hidden", String(!value));
    this.toggle.setAttribute("aria-expanded", String(value));
    document.body.classList.toggle("has-tasks", value);
  }
  update(works) {
    const isNew = works.some((work) => !this.items.has(work.work_id));
    for (const work of works) this.items.set(work.work_id, work);
    this.active = true;
    this.render();
    if (isNew) {
      this.show(true);
      this.toggle.classList.remove("pulse");
      void this.toggle.offsetWidth;
      this.toggle.classList.add("pulse");
    }
  }
  clear() {
    this.items.clear();
    this.nodes.clear();
    this.list.replaceChildren();
    this.toggle.hidden = true;
    this.active = false;
    this.show(false);
  }
  end() {
    this.active = false;
    this.endedAt = Date.now() / 1000;
    this.render();
  }
  state(work) {
    if (work.state === "delivered") return "taskDelivered";
    if (work.state === "failed") return "taskFailed";
    if (work.state === "cancelled") return "taskCancelled";
    if (work.state === "cancelling") return "taskCancelling";
    if (
      ["polish", "completed", "delivering"].includes(work.phase) ||
      ["completed", "delivering"].includes(work.state)
    )
      return "taskReturning";
    if (work.phase === "routing" || work.capability === "unrouted")
      return "taskRouting";
    if (work.state === "queued") return "taskQueued";
    return "taskRunning";
  }
  render() {
    this.toggle.hidden = !this.items.size;
    document.getElementById("task-count").textContent = String(
      [...this.items.values()].filter((work) => !terminal.has(work.state))
        .length || this.items.size,
    );
    const ordered = [...this.items.values()].sort(
      (a, b) => b.created_at - a.created_at,
    );
    for (const work of ordered) {
      let card = this.nodes.get(work.work_id);
      if (!card) {
        card = element("article", "task-card");
        card.dataset.workId = work.work_id;
        const head = element("div", "task-card-header"),
          state = element("span", "task-state");
        state.append(
          element("span", "task-orbit"),
          element("span", "task-state-label"),
        );
        head.append(state, element("time", "task-elapsed"));
        card.append(
          head,
          element("h3", "task-objective"),
          element("p", "task-progress"),
        );
        const track = element("div", "task-track");
        track.setAttribute("aria-hidden", "true");
        track.append(element("span"));
        card.append(track);
        const actions = element("div", "task-actions");
        actions.append(
          element("span", "task-stage"),
          element("button", "task-cancel"),
        );
        card.append(actions, element("div", "task-files"));
        this.nodes.set(work.work_id, card);
        this.list.prepend(card);
      }
      const done = work.state === "delivered",
        ended = terminal.has(work.state),
        state = this.active || ended ? this.state(work) : "conversationEnded";
      card.classList.toggle("inactive", !this.active);
      card.classList.toggle("done", done);
      card.classList.toggle("failed", work.state === "failed");
      card.classList.toggle("cancelled", work.state === "cancelled");
      card.querySelector(".task-state-label").textContent = t(state);
      const orbit = card.querySelector(".task-orbit");
      orbit.replaceChildren();
      if (done) orbit.append(icon("check"));
      card.querySelector(".task-objective").textContent = work.objective;
      const records = work.feedback || [],
        latest = records.length ? records[records.length - 1].text : "";
      card.querySelector(".task-progress").textContent =
        work.error_message ||
        work.result_text ||
        work.progress_text ||
        latest ||
        t(
          work.state === "cancelled"
            ? "taskCancelledHint"
            : state === "taskRouting"
              ? "taskPreparing"
              : state === "taskReturning"
                ? "taskReturnHint"
                : "taskDefault",
        );
      card.querySelector(".task-stage").textContent = this.active
        ? t(state)
        : t("conversationEnded");
      const cancel = card.querySelector(".task-cancel");
      cancel.textContent = t(
        work.state === "cancelling" ? "taskCancelling" : "cancel",
      );
      cancel.hidden = ended || !this.active;
      cancel.disabled = work.state === "cancelling";
      cancel.onclick = () => {
        cancel.disabled = true;
        cancel.textContent = t("taskCancelling");
        this.send({ type: "cancel_work", work_id: work.work_id });
      };
      const files = card.querySelector(".task-files");
      const signature = JSON.stringify(work.artifacts || []) + this.active;
      if (files.dataset.signature !== signature) {
        files.dataset.signature = signature;
        files.replaceChildren();
        for (const artifact of work.artifacts || []) {
          const button = element("button", "artifact-button");
          button.append(icon("download"), element("span", "", artifact.name));
          button.title = t("download");
          button.disabled = !this.active;
          button.onclick = () => this.download(work.work_id, artifact);
          files.append(button);
        }
      }
    }
    this.updateTime();
  }
  updateTime() {
    for (const [id, card] of this.nodes) {
      const work = this.items.get(id),
        end =
          work.completed_at ||
          (!this.active && this.endedAt) ||
          Date.now() / 1000,
        seconds = Math.max(0, Math.floor(end - work.created_at));
      card.querySelector("time").textContent =
        `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    }
  }
}
