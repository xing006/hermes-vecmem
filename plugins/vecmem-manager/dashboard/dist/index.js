(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK) return;

  const I18N = {
    en: {
      title: "VecMem Manager", refresh: "Refresh", loading: "Loading...",
      no_topic: "no topic", close: "Close", detail: "Detail",
      archive: "Archive", restore: "Restore", approve: "Approve", reject: "Reject",
      search: "Search...", none: "None", ok: "OK", fail: "FAIL",
      actions: "Actions", total: "Total", active: "Active",
      review: "Review", archived: "Archived", superseded: "superseded",
      db_size: "DB Size", consistency: "Consistency",
      filter_status: "Status", filter_category: "Category",
      filter_type: "Type", filter_topic: "Topic",
      all: "all", any: "Any", category: "category",
      type: "type", topic: "Topic",
      content: "Content", conf: "Conf", updated: "Updated",
      detail_title: "#{id} Details", decision_reason: "decision_reason:",
      same_topic: "Same topic records", event_history: "Event history",
      showing: "Showing {count} / {total}", prev: "\u2190 Prev", next: "Next \u2192",
    },
    "zh-CN": {
      title: "VecMem 管理", refresh: "刷新", loading: "加载中...",
      no_topic: "无主题", close: "关闭", detail: "详情",
      archive: "归档", restore: "恢复", approve: "通过", reject: "驳回",
      search: "搜索...", none: "无", ok: "正常", fail: "异常",
      actions: "操作", total: "总计", active: "活跃",
      review: "待审", archived: "已归档", superseded: "已废弃",
      db_size: "数据库大小", consistency: "一致性",
      filter_status: "状态", filter_category: "分类",
      filter_type: "类型", filter_topic: "主题",
      all: "全部", any: "不限", category: "分类",
      type: "类型", topic: "主题",
      content: "内容", conf: "置信度", updated: "更新时间",
      detail_title: "#{id} 详情", decision_reason: "决策原因：",
      same_topic: "同主题记录", event_history: "事件历史",
      showing: "显示 {count} / {total} 条", prev: "\u2190 上一页", next: "下一页 \u2192",
    },
  };

  function makeTranslator(locale) {
    var pack = I18N[locale] || I18N.en;
    return function t(key, params) {
      var template = pack[key] || I18N.en[key] || key;
      if (!params) return template;
      return Object.keys(params).reduce(function (s, k) { return s.replace("{" + k + "}", params[k]); }, template);
    };
  }

  function detectLocale() {
    var raw = String(navigator.language || "en");
    return raw.toLowerCase().startsWith("zh") ? "zh-CN" : "en";
  }

  var locale = detectLocale();
  var t = makeTranslator(locale);
  var h = SDK.React.createElement;
  var useState = SDK.hooks.useState;
  var useEffect = SDK.hooks.useEffect;
  var useCallback = SDK.hooks.useCallback;
  var fetchJSON = SDK.fetchJSON;
  var BASE = "/api/plugins/vecmem-manager";
  var PAGE_SIZE = 100;

  function fmtTime(ts) {
    if (!ts) return "\u2014";
    var d = new Date(Number(ts) * 1000);
    if (Number.isNaN(d.getTime())) return "\u2014";
    var pad = function (n) { return String(n).padStart(2, "0"); };
    return (d.getMonth() + 1) + "/" + pad(d.getDate()) + " " + pad(d.getHours()) + ":" + pad(d.getMinutes());
  }

  function fmtBytes(n) {
    n = Number(n || 0);
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / 1024 / 1024).toFixed(1) + " MB";
  }

  function statusClass(s) {
    return "vm-badge vm-status-" + (s || "active");
  }

  function toast(msg, type) {
    var el = document.createElement("div");
    el.className = "vm-toast vm-toast-" + (type || "success");
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(function () { el.remove(); }, 2600);
  }

  function StatCard(props) {
    return h("div", { className: "vm-card " + (props.kind || "") },
      h("div", { className: "vm-card-label" }, props.label),
      h("div", { className: "vm-card-value" }, props.value == null ? "\u2014" : props.value),
      props.sub ? h("div", { className: "vm-card-sub" }, props.sub) : null
    );
  }

  function FilterSelect(props) {
    var options = props.options || [];
    var label = props.label || "";
    var _a = useState(false), open = _a[0], setOpen = _a[1];
    var ref = SDK.React.useRef(null);

    useEffect(function () {
      if (!open) return;
      function handler(e) {
        if (ref.current && !ref.current.contains(e.target)) {
          setOpen(false);
        }
      }
      document.addEventListener("mousedown", handler);
      return function () { document.removeEventListener("mousedown", handler); };
    }, [open]);

    var currentLabel = "";
    if (props.value) {
      var matched = options.filter(function (o) { return (o.value || o.category || o.memory_type || o.topic_key) === props.value; });
      if (matched.length) {
        currentLabel = (matched[0].value || matched[0].category || matched[0].memory_type || matched[0].topic_key) + " (" + matched[0].cnt + ")";
      }
    }
    // Show label as prefix hint when nothing selected
    var triggerText = props.value ? currentLabel : (label ? label : t("any"));

    return h("div", { className: "vm-custom-select", ref: ref },
      h("div", {
        className: "vm-custom-select-trigger",
        onClick: function () { setOpen(!open); },
        onKeyDown: function (e) { if (e.key === "Enter") { setOpen(!open); } },
        tabIndex: 0,
        role: "combobox",
        "aria-expanded": open,
      },
        h("span", { className: "vm-custom-select-text" + (props.value ? "" : " vm-custom-select-placeholder") }, triggerText),
        h("span", { className: "vm-custom-select-arrow" }, open ? "\u25B2" : "\u25BC")
      ),
      open ? h("div", { className: "vm-custom-select-menu" },
        h("div", {
          className: "vm-custom-select-option" + (!props.value ? " selected" : ""),
          onClick: function () { props.onChange(""); setOpen(false); },
        }, label ? label : t("any")),
        options.map(function (o) {
          var val = o.value || o.category || o.memory_type || o.topic_key || "";
          var isSelected = val === props.value;
          return h("div", {
            key: val,
            className: "vm-custom-select-option" + (isSelected ? " selected" : ""),
            onClick: function () { props.onChange(val); setOpen(false); },
          }, (val) + " (" + o.cnt + ")");
        })
      ) : null
    );
  }

  function DetailPanel(props) {
    if (!props.detail) return null;
    var tt = props.t;
    var r = props.detail.record;
    return h("div", { className: "vm-detail" },
      h("div", { className: "vm-detail-head" },
        h("div", null,
          h("div", { className: "vm-title" }, tt("detail_title", { id: r.id })),
          h("div", { className: "vm-muted" }, r.topic_key || tt("no_topic"))
        ),
        h("button", { className: "vm-btn", onClick: props.onClose }, tt("close"))
      ),
      h("pre", { className: "vm-content" }, r.content || ""),
      h("div", { className: "vm-meta" },
        h("span", { className: statusClass(r.status) }, r.status || "active"),
        h("span", null, tt("category") + ": " + (r.category || "\u2014")),
        h("span", null, tt("type") + ": " + (r.memory_type || "\u2014")),
        h("span", null, tt("conf") + ": " + (r.confidence == null ? "\u2014" : r.confidence))
      ),
      h("div", { className: "vm-reason" },
        h("b", null, tt("decision_reason") + " "), r.decision_reason || "\u2014"
      ),
      h("div", { className: "vm-actions" },
        h("button", { className: "vm-btn", onClick: function () { props.onAction(r.id, "archive"); } }, tt("archive")),
        h("button", { className: "vm-btn", onClick: function () { props.onAction(r.id, "restore"); } }, tt("restore")),
        h("button", { className: "vm-btn", onClick: function () { props.onAction(r.id, "approve"); } }, tt("approve")),
        h("button", { className: "vm-btn vm-btn-danger", onClick: function () { props.onAction(r.id, "reject"); } }, tt("reject"))
      ),
      h("div", { className: "vm-section-title" }, tt("same_topic")),
      props.detail.same_topic_records && props.detail.same_topic_records.length
        ? h("ul", { className: "vm-list" }, props.detail.same_topic_records.map(function (x) { return h("li", { key: x.id }, "#" + x.id + " [" + x.status + "] " + x.content); }))
        : h("div", { className: "vm-empty" }, tt("none")),
      h("div", { className: "vm-section-title" }, tt("event_history")),
      props.detail.events && props.detail.events.length
        ? h("ul", { className: "vm-list" }, props.detail.events.map(function (e) {
            return h("li", { key: e.event_id }, fmtTime(e.created_at) + " " + e.action + ": " + (e.before_status || "\u2014") + " \u2192 " + (e.after_status || "\u2014") + " (" + (e.reason || "") + ")");
          }))
        : h("div", { className: "vm-empty" }, tt("none"))
    );
  }

  function VecMemManager() {
    var _a = useState(null), stats = _a[0], setStats = _a[1];
    var _b = useState([]), records = _b[0], setRecords = _b[1];
    var _c = useState(0), total = _c[0], setTotal = _c[1];
    var _d = useState(true), loading = _d[0], setLoading = _d[1];
    var _e = useState(null), error = _e[0], setError = _e[1];
    var _f = useState(null), detail = _f[0], setDetail = _f[1];
    var _g = useState(null), choices = _g[0], setChoices = _g[1];
    var _h = useState({ status: "active", category: "", memory_type: "", topic_key: "", q: "", offset: 0 }), filters = _h[0], setFilters = _h[1];

    var query = useCallback(function () {
      var p = new URLSearchParams();
      Object.keys(filters).forEach(function (k) { if (filters[k]) p.set(k, filters[k]); });
      p.set("limit", PAGE_SIZE);
      return p.toString();
    }, [filters]);

    var refresh = useCallback(function () {
      setLoading(true);
      Promise.all([
        fetchJSON(BASE + "/stats"),
        fetchJSON(BASE + "/records?" + query()),
      ]).then(function (results) {
        setStats(results[0]);
        setRecords(results[1].records || []);
        setTotal(results[1].total || 0);
        setError(null);
        setLoading(false);
      }).catch(function (e) {
        setError(e.message || String(e));
        setLoading(false);
      });
    }, [query]);

    useEffect(function () { refresh(); }, [refresh]);

    useEffect(function () {
      fetchJSON(BASE + "/choices").then(function (data) {
        setChoices(data);
      }).catch(function () {});
    }, []);

    function loadDetail(id) {
      fetchJSON(BASE + "/records/" + id).then(function (data) { setDetail(data); }).catch(function (e) { toast(e.message, "error"); });
    }

    function doAction(id, action) {
      var reasonLabels = locale === "zh-CN"
        ? { archive: "已归档", restore: "已恢复", approve: "已通过", reject: "已驳回" }
        : { archive: "Archived", restore: "Restored", approve: "Approved", reject: "Rejected" };
      fetchJSON(BASE + "/records/" + id + "/" + action, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: "dashboard " + action }),
      }).then(function () {
        toast(reasonLabels[action] || action + " ok");
        setDetail(null);
        refresh();
      }).catch(function (e) { toast(t("fail") + ": " + e.message, "error"); });
    }

    function setFilter(k, v) {
      var next = {};
      Object.keys(filters).forEach(function (key) { next[key] = filters[key]; });
      next[k] = v;
      if (k !== "offset") next.offset = 0;
      setFilters(next);
    }

    function goPage(dir) {
      var next = {};
      Object.keys(filters).forEach(function (key) { next[key] = filters[key]; });
      next.offset = Math.max(0, filters.offset + dir * PAGE_SIZE);
      setFilters(next);
    }

    var page = Math.floor(filters.offset / PAGE_SIZE) + 1;
    var totalPages = Math.ceil(total / PAGE_SIZE) || 1;

    var statusOptions = [
      { value: "active", cnt: stats && stats.by_status && (stats.by_status.active || 0) },
      { value: "review", cnt: stats && stats.by_status && (stats.by_status.review || 0) },
      { value: "archived", cnt: stats && stats.by_status && (stats.by_status.archived || 0) },
      { value: "superseded", cnt: stats && stats.by_status && (stats.by_status.superseded || 0) },
      { value: "all", cnt: stats && stats.total },
    ];

    return h("div", { className: "vm-wrap" },
      h("div", { className: "vm-header" },
        h("div", null,
          h("div", { className: "vm-title" }, t("title")),
          h("div", { className: "vm-muted" }, stats ? stats.db_path : t("loading"))
        ),
        h("button", { className: "vm-primary", onClick: refresh, disabled: loading }, loading ? t("loading") : t("refresh"))
      ),
      error ? h("div", { className: "vm-error" }, error) : null,
      h("div", { className: "vm-grid" },
        h(StatCard, { label: t("total"), value: stats && stats.total }),
        h(StatCard, { label: t("active"), value: stats && stats.active }),
        h(StatCard, { label: t("review"), value: stats && stats.by_status && (stats.by_status.review || 0), kind: stats && stats.by_status && stats.by_status.review ? "warn" : "" }),
        h(StatCard, { label: t("archived"), value: stats && stats.by_status && (stats.by_status.archived || 0) }),
        h(StatCard, { label: t("db_size"), value: stats && fmtBytes(stats.db_size_bytes) }),
        h(StatCard, { label: t("consistency"), value: stats && stats.consistency && stats.consistency.ok ? t("ok") : t("fail"), kind: stats && stats.consistency && stats.consistency.ok ? "ok" : "danger" })
      ),
      h("div", { className: "vm-filters" },
        h(FilterSelect, {
          label: t("filter_status"),
          options: statusOptions,
          value: filters.status,
          onChange: function (v) { setFilter("status", v || "active"); },
        }),
        h(FilterSelect, {
          label: t("filter_category"),
          options: choices && choices.categories || [],
          value: filters.category,
          onChange: function (v) { setFilter("category", v); },
        }),
        h(FilterSelect, {
          label: t("filter_type"),
          options: choices && choices.memory_types || [],
          value: filters.memory_type,
          onChange: function (v) { setFilter("memory_type", v); },
        }),
        h(FilterSelect, {
          label: t("filter_topic"),
          options: choices && choices.topic_keys || [],
          value: filters.topic_key,
          onChange: function (v) { setFilter("topic_key", v); },
        }),
        h("input", { placeholder: t("search"), value: filters.q, onChange: function (e) { setFilter("q", e.target.value); } })
      ),
      h("div", { className: "vm-table-wrap" },
        h("div", { className: "vm-table-head" },
          h("span", null, t("showing", { count: records.length, total: total })),
          h("div", { className: "vm-page-actions" },
            h("button", { className: "vm-btn", disabled: filters.offset <= 0, onClick: function () { goPage(-1); } }, t("prev")),
            h("span", { className: "vm-page-num" }, page + " / " + totalPages),
            h("button", { className: "vm-btn", disabled: filters.offset + PAGE_SIZE >= total, onClick: function () { goPage(1); } }, t("next"))
          )
        ),
        h("table", { className: "vm-table" },
          h("thead", null, h("tr", null,
            ["ID", "Status", t("category"), t("type"), t("topic"), t("content"), t("conf"), t("updated"), t("actions")].map(function (x) { return h("th", { key: x }, x); })
          )),
          h("tbody", null, records.map(function (r) {
            return h("tr", { key: r.id },
              h("td", null, r.id),
              h("td", null, h("span", { className: statusClass(r.status) }, r.status || "active")),
              h("td", null, r.category || "\u2014"),
              h("td", null, r.memory_type || "\u2014"),
              h("td", { className: "vm-topic" }, r.topic_key || "\u2014"),
              h("td", { className: "vm-text" }, r.content || ""),
              h("td", null, r.confidence == null ? "\u2014" : Number(r.confidence).toFixed(2)),
              h("td", null, fmtTime(r.updated_at)),
              h("td", { className: "vm-row-actions" },
                h("button", { className: "vm-btn", onClick: function () { loadDetail(r.id); } }, t("detail")),
                h("button", { className: "vm-btn", onClick: function () { doAction(r.id, "archive"); } }, t("archive")),
                h("button", { className: "vm-btn", onClick: function () { doAction(r.id, "restore"); } }, t("restore"))
              )
            );
          }))
        )
      ),
      h(DetailPanel, { detail: detail, onClose: function () { setDetail(null); }, onAction: doAction, t: t })
    );
  }

  window.__HERMES_PLUGINS__.register("vecmem-manager", VecMemManager);
})();
