"use strict";

const elements = {
  form: document.querySelector("#search-form"),
  query: document.querySelector("#query-input"),
  topK: document.querySelector("#top-k-input"),
  strategy: document.querySelector("#strategy-input"),
  submit: document.querySelector("#submit-button"),
  message: document.querySelector("#message"),
  health: document.querySelector("#health-status"),
  answerSection: document.querySelector("#answer-section"),
  answerText: document.querySelector("#answer-text"),
  citations: document.querySelector("#citation-links"),
  limitations: document.querySelector("#limitations-list"),
  modeChip: document.querySelector("#mode-chip"),
  evidenceSection: document.querySelector("#evidence-section"),
  resultCount: document.querySelector("#result-count"),
  results: document.querySelector("#results-grid"),
  debugPanel: document.querySelector("#debug-panel"),
  debugList: document.querySelector("#debug-list"),
  understandingSection: document.querySelector("#understanding-section"),
  understandingList: document.querySelector("#understanding-list"),
  languageChip: document.querySelector("#language-chip"),
  intentNotice: document.querySelector("#intent-notice"),
};

const FALLBACK_LABELS = {
  not_configured: "未配置模型",
  timeout: "生成超时",
  provider_error: "供应商错误",
  model_error: "模型错误",
  invalid_output: "生成结果未通过校验",
};

const FIELD_LABELS = {
  product_name: "商品名称",
  product_class: "商品类别",
  category_hierarchy: "类别层级",
  product_description: "商品描述",
  product_features: "商品特征",
  rating_count: "评分数量",
  average_rating: "平均评分",
  review_count: "评论数量",
};

const LANGUAGE_LABELS = { zh: "中文", en: "英文", mixed: "中英混合", other: "其他" };
const INTENT_LABELS = {
  price: "当前价格",
  discount: "折扣或促销",
  inventory: "实时库存",
  delivery: "配送信息",
  return_policy: "退换货政策",
  warranty: "保修或售后",
  review_text: "评论正文",
};
const REWRITE_LABELS = {
  original: "原始英文查询",
  rules: "本地规则",
  llm: "模型结构化改写",
  rules_fallback: "模型失败，已使用本地规则",
};

function setText(node, value) {
  node.textContent = value == null || value === "" ? "数据未提供" : String(value);
}

function createElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function clearNode(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    elements.health.textContent = payload.index_ready
      ? `${payload.product_count.toLocaleString()} 件商品 · 索引就绪`
      : "索引未就绪";
    const vectorOption = elements.strategy.querySelector("option[value='vector']");
    vectorOption.disabled = !payload.vector_index_ready;
    vectorOption.textContent = payload.vector_index_ready
      ? "Vector（v0.2）"
      : "Vector（v0.2 · 索引未就绪）";
    const hybridOption = elements.strategy.querySelector("option[value='hybrid']");
    const rerankOption = elements.strategy.querySelector("option[value='rerank']");
    hybridOption.disabled = !payload.vector_index_ready;
    rerankOption.disabled = !payload.vector_index_ready || !payload.reranker_ready;
    hybridOption.textContent = payload.vector_index_ready
      ? "Hybrid RRF（v0.3）"
      : "Hybrid RRF（v0.3 · 索引未就绪）";
    rerankOption.textContent = payload.vector_index_ready && payload.reranker_ready
      ? "Hybrid + Reranker（v0.3）"
      : "Hybrid + Reranker（v0.3 · 模型未就绪）";
  } catch (_) {
    elements.health.textContent = "无法检查服务状态";
  }
}

function selectedMode() {
  return document.querySelector("input[name='mode']:checked").value;
}

function updateSubmitLabel() {
  elements.submit.textContent = selectedMode() === "search" ? "只检索" : "搜索并回答";
}

function setLoading(loading) {
  elements.submit.disabled = loading;
  elements.query.disabled = loading;
  elements.topK.disabled = loading;
  elements.strategy.disabled = loading;
  elements.submit.textContent = loading ? "正在检索…" : "";
  document.querySelectorAll("input[name='mode']").forEach((input) => {
    input.disabled = loading;
  });
  if (!loading) updateSubmitLabel();
}

function displayError(error) {
  const requestId = error.request_id ? `（错误 ID：${error.request_id}）` : "";
  const message = error.error?.message || error.message || "请求失败，请稍后重试。";
  elements.message.textContent = `${message}${requestId}`;
}

function addDefinition(term, value) {
  elements.debugList.append(createElement("dt", "", term), createElement("dd", "", value));
}

function renderDebug(payload, topK) {
  clearNode(elements.debugList);
  const timing = payload.timing || { retrieval_ms: payload.latency_ms, total_ms: payload.latency_ms };
  addDefinition("请求 ID", payload.request_id || "数据未提供");
  addDefinition("标准化查询", payload.normalized_query || payload.query);
  addDefinition("查询理解耗时", `${Number(timing.rewrite_ms || 0).toFixed(3)} ms`);
  addDefinition("Top-K", topK);
  addDefinition("检索策略", payload.retrieval_strategy || "hybrid");
  addDefinition("检索耗时", `${Number(timing.retrieval_ms || 0).toFixed(3)} ms`);
  addDefinition("证据构造耗时", `${Number(timing.evidence_ms || 0).toFixed(3)} ms`);
  addDefinition("生成耗时", `${Number(timing.generation_ms || 0).toFixed(3)} ms`);
  addDefinition("总耗时", `${Number(timing.total_ms || payload.latency_ms || 0).toFixed(3)} ms`);
  elements.debugPanel.classList.remove("hidden");
}

function renderUnderstanding(payload) {
  const understanding = payload.query_understanding;
  if (!understanding) return;
  clearNode(elements.understandingList);
  const addItem = (term, value) => {
    elements.understandingList.append(
      createElement("dt", "", term),
      createElement("dd", "", value || "未识别"),
    );
  };
  addItem("英文检索表达", understanding.retrieval_query);
  addItem("查询处理方式", REWRITE_LABELS[understanding.rewrite_source] || understanding.rewrite_source);
  addItem("识别属性", JSON.stringify(understanding.attributes || {}));
  addItem("排除条件", (understanding.excluded_terms || []).join("、") || "未识别");
  elements.languageChip.textContent = LANGUAGE_LABELS[understanding.detected_language] || "未知语言";
  const unavailable = (understanding.unsupported_intents || []).map(
    (item) => INTENT_LABELS[item] || item,
  );
  if (unavailable.length) {
    elements.intentNotice.textContent = `你还询问了${unavailable.join("、")}；当前数据不包含这些信息，系统不会编造。`;
    elements.intentNotice.classList.remove("hidden");
  } else {
    elements.intentNotice.classList.add("hidden");
  }
  elements.understandingSection.classList.remove("hidden");
}

function renderCard(product) {
  const card = createElement("article", "product-card");
  card.id = `product-${product.citation_id}`;
  card.tabIndex = -1;
  const top = createElement("div", "card-top");
  const titleGroup = createElement("div");
  titleGroup.append(
    createElement("div", "card-rank", `排名 #${product.rank} · 商品 ID ${product.product_id}`),
    createElement("h3", "card-title", product.product_name),
  );
  top.append(titleGroup, createElement("span", "citation-badge", `[${product.citation_id}]`));

  const meta = createElement("p", "card-meta");
  setText(meta, product.product_class || product.category_hierarchy);
  const description = createElement("p", "card-description");
  setText(description, product.product_description);
  const features = createElement("div", "card-features");
  setText(features, product.product_features);

  const stats = createElement("div", "card-stats");
  stats.append(
    createElement("span", "", `分数 ${Number(product.score).toFixed(4)}`),
    createElement("span", "", `平均评分 ${product.average_rating ?? "数据未提供"}`),
    createElement("span", "", `评分数 ${product.rating_count ?? "数据未提供"}`),
    createElement("span", "", `评论数 ${product.review_count ?? "数据未提供"}`),
  );
  if ((product.retrieval_sources || []).length) {
    stats.append(
      createElement(
        "span",
        "",
        `来源 ${product.retrieval_sources.join("+")} ${JSON.stringify(product.source_ranks || {})}`,
      ),
    );
  }
  const matched = createElement("div", "matched-list");
  (product.matched_fields || []).forEach((field) => {
    matched.append(createElement("span", "", `命中：${FIELD_LABELS[field] || field}`));
  });
  const link = createElement("a", "product-link", "通过本地商品接口核验原始字段 →");
  link.href = `/api/products/${product.product_id}`;
  link.target = "_blank";
  link.rel = "noopener";

  card.append(top, meta, description, features, stats, matched, link);
  return card;
}

function focusCitation(citationId) {
  document.querySelectorAll(".product-card.highlighted").forEach((node) => {
    node.classList.remove("highlighted");
  });
  const target = document.querySelector(`#product-${CSS.escape(citationId)}`);
  if (!target) return;
  target.classList.add("highlighted");
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  target.focus({ preventScroll: true });
}

function renderResults(results) {
  clearNode(elements.results);
  results.forEach((product) => elements.results.append(renderCard(product)));
  elements.resultCount.textContent = `${results.length} 个结果`;
  elements.evidenceSection.classList.remove("hidden");
}

function renderAnswer(payload) {
  clearNode(elements.citations);
  clearNode(elements.limitations);
  if (payload.answer) {
    elements.answerText.textContent = payload.answer;
  } else {
    elements.answerText.textContent = "只检索模式：请直接核验下方商品证据。";
  }
  const modeLabel = payload.mode === "rag" ? `RAG · ${payload.model}` : "只检索模式";
  const fallback = payload.fallback_reason ? ` · ${FALLBACK_LABELS[payload.fallback_reason]}` : "";
  elements.modeChip.textContent = `${modeLabel}${fallback}`;
  (payload.citations || []).forEach((citation) => {
    const button = createElement("button", "citation-link", `[${citation.citation_id}]`);
    button.type = "button";
    button.addEventListener("click", () => focusCitation(citation.citation_id));
    elements.citations.append(button);
  });
  (payload.limitations || []).forEach((limitation) => {
    elements.limitations.append(createElement("li", "", limitation));
  });
  elements.answerSection.classList.remove("hidden");
}

async function submitQuery(event) {
  event.preventDefault();
  elements.message.textContent = "";
  elements.answerSection.classList.add("hidden");
  elements.evidenceSection.classList.add("hidden");
  elements.understandingSection.classList.add("hidden");
  elements.debugPanel.classList.add("hidden");
  setLoading(true);
  const mode = selectedMode();
  const topK = Number(elements.topK.value);
  const retrievalStrategy = elements.strategy.value;
  try {
    const response = await fetch(mode === "answer" ? "/api/answer" : "/api/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        query: elements.query.value,
        top_k: topK,
        retrieval_strategy: retrievalStrategy,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw payload;
    renderUnderstanding(payload);
    renderAnswer(payload);
    renderResults(payload.results || []);
    renderDebug(payload, topK);
    if (!(payload.results || []).length) {
      elements.message.textContent = "没有检索到匹配商品，请尝试简化需求或更换关键词。";
    }
  } catch (error) {
    displayError(error);
  } finally {
    setLoading(false);
  }
}

elements.form.addEventListener("submit", submitQuery);
elements.query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});
document.querySelectorAll(".example-query").forEach((button) => {
  button.addEventListener("click", () => {
    elements.query.value = button.textContent;
    elements.query.focus();
  });
});
document.querySelectorAll("input[name='mode']").forEach((input) => {
  input.addEventListener("change", updateSubmitLabel);
});

checkHealth();
