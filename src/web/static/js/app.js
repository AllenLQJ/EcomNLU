/**
 * 电商评论智能分析系统 · 前端逻辑
 *
 * 约束（docs/web-development-spec.md）：
 * - 只调用契约 §3.1 的三个端点，不引入第二套接口或预测逻辑。
 * - 所有来自接口或用户的文本一律用 textContent 写入 DOM，不使用 innerHTML。
 * - 不使用 eval、内联事件处理器或动态执行字符串。
 * - 页面不推断情感；sentiment 为 null 时如实显示「尚未启用」。
 *
 * 纯函数集中挂在 window.WebApp 上，便于在浏览器控制台或后续测试中单独验证
 * （说明书 §3.1 要求核心状态转换可测试）。
 */

'use strict';

/* ==========================================================================
   常量
   ========================================================================== */

var ENDPOINTS = {
  health: '/api/health',
  categories: '/api/categories',
  predict: '/api/predict'
};

var REQUEST_TIMEOUT_MS = 15000;
var TOAST_DURATION_MS = 3200;

/** 建议处理方式：前端确定性映射，**不是模型预测**（说明书 §4.7）。 */
var ROUTING_BY_CATEGORY = {
  '书籍': '转交图书售后团队',
  '平板': '转交数码产品售后团队',
  '手机': '转交手机售后团队',
  '水果': '转交生鲜售后团队',
  '洗发水': '转交个护售后团队',
  '热水器': '转交家电售后团队',
  '蒙牛': '转交食品饮料售后团队',
  '衣服': '转交服装售后团队',
  '计算机': '转交计算机售后团队',
  '酒店': '转交酒店服务团队'
};

var FALLBACK_ROUTING = '转交人工客服团队';

/** 契约 §3.4 错误码到界面文案（说明书 §7.4）。 */
var ERROR_TEXT_BY_CODE = {
  INVALID_JSON: '请求格式错误，请刷新页面后重试',
  MISSING_REVIEW: '请输入评论内容',
  EMPTY_REVIEW: '评论内容不能为空',
  REVIEW_TOO_LONG: '评论超过长度上限，请精简后重试',
  INVALID_TOP_K: 'Top-K 必须是 1 到 10 的整数',
  MODEL_NOT_LOADED: '模型尚未加载，请用 --model-dir 指定模型产物后重启服务',
  INTERNAL_ERROR: '服务内部错误，请稍后重试'
};

var GENERIC_ERROR_TEXT = '请求失败，请稍后重试';

/* ==========================================================================
   纯函数（可单独测试）
   ========================================================================== */

/** 把 0–1 的概率格式化为一位小数的百分比；非法值返回 null。 */
function formatPercent(value) {
  if (typeof value !== 'number' || !isFinite(value)) return null;
  if (value < 0 || value > 1) return null;
  return (value * 100).toFixed(1) + '%';
}

/** 把毫秒数格式化为 `12.4 ms`；非法值返回 null。 */
function formatElapsedMs(value) {
  if (typeof value !== 'number' || !isFinite(value) || value < 0) return null;
  return value.toFixed(1) + ' ms';
}

/** 把概率裁剪到 0–1，用于概率条宽度，避免非法值撑坏样式。 */
function clampRatio(value) {
  if (typeof value !== 'number' || !isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

/** 建议处理方式与优先级；未知类别回退到通用建议，不猜测。 */
function getRoutingAdvice(categoryLabel, sentiment) {
  var advice = Object.prototype.hasOwnProperty.call(ROUTING_BY_CATEGORY, categoryLabel)
    ? ROUTING_BY_CATEGORY[categoryLabel]
    : FALLBACK_ROUTING;
  var negative = !!(sentiment && sentiment.label === '负向');
  return {
    advice: advice,
    priority: negative ? '优先级：较高' : '优先级：普通',
    urgent: negative
  };
}

/**
 * 校验 /api/predict 响应结构。结构不合法时不得渲染，避免展示半截数据。
 * 返回 {ok, reason}。
 */
function validatePredictResponse(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    return { ok: false, reason: '响应不是 JSON 对象' };
  }
  var category = body.category;
  if (!category || typeof category !== 'object') {
    return { ok: false, reason: '缺少 category' };
  }
  if (typeof category.label !== 'string' || !category.label) {
    return { ok: false, reason: 'category.label 非法' };
  }
  if (formatPercent(category.confidence) === null) {
    return { ok: false, reason: 'category.confidence 非法' };
  }
  if (!Array.isArray(body.top_k)) {
    return { ok: false, reason: 'top_k 不是数组' };
  }
  for (var i = 0; i < body.top_k.length; i++) {
    var item = body.top_k[i];
    if (!item || typeof item !== 'object') return { ok: false, reason: 'top_k 项非法' };
    if (typeof item.label !== 'string' || !item.label) {
      return { ok: false, reason: 'top_k 项缺少 label' };
    }
    if (formatPercent(item.probability) === null) {
      return { ok: false, reason: 'top_k 项 probability 非法' };
    }
  }
  if (formatElapsedMs(body.elapsed_ms) === null) {
    return { ok: false, reason: 'elapsed_ms 非法' };
  }
  if (typeof body.model_name !== 'string' || !body.model_name) {
    return { ok: false, reason: 'model_name 非法' };
  }
  if (body.sentiment !== null && body.sentiment !== undefined) {
    var sentiment = body.sentiment;
    if (typeof sentiment !== 'object' || typeof sentiment.label !== 'string') {
      return { ok: false, reason: 'sentiment 非法' };
    }
    if (sentiment.label !== '正向' && sentiment.label !== '负向') {
      return { ok: false, reason: 'sentiment.label 只能是正向或负向' };
    }
    if (formatPercent(sentiment.confidence) === null) {
      return { ok: false, reason: 'sentiment.confidence 非法' };
    }
  }
  return { ok: true, reason: '' };
}

/** 判定服务状态：ok / degraded / error（说明书 §4.1）。 */
function classifyHealth(status, body) {
  if (!body || typeof body !== 'object') return 'error';
  if (status === 200 && body.status === 'ok') return 'ok';
  if (status === 503 && body.model_loaded === false) return 'degraded';
  if (typeof body.status === 'string' && body.model_loaded === true) return 'ok';
  return 'error';
}

/**
 * 错误码到界面文案（说明书 §7.4）。
 *
 * 优先使用 API 返回的 message——它带有长度、上限等具体信息，比固定文案更有用；
 * 接口未给 message 时回退到错误码表，未知错误码回退到通用文案，不暴露内部细节。
 */
function mapErrorText(code, apiMessage) {
  if (typeof apiMessage === 'string' && apiMessage.trim()) {
    return apiMessage.trim();
  }
  if (Object.prototype.hasOwnProperty.call(ERROR_TEXT_BY_CODE, code)) {
    return ERROR_TEXT_BY_CODE[code];
  }
  return GENERIC_ERROR_TEXT;
}

/* 供测试与调试使用。 */
window.WebApp = {
  formatPercent: formatPercent,
  formatElapsedMs: formatElapsedMs,
  clampRatio: clampRatio,
  getRoutingAdvice: getRoutingAdvice,
  validatePredictResponse: validatePredictResponse,
  classifyHealth: classifyHealth,
  mapErrorText: mapErrorText,
  ROUTING_BY_CATEGORY: ROUTING_BY_CATEGORY
};

/* ==========================================================================
   DOM 引用
   ========================================================================== */

var dom = {};

function cacheDom() {
  var ids = [
    'service-status', 'service-status-text', 'btn-help', 'btn-about', 'btn-settings',
    'predict-form', 'review', 'char-count', 'top-k', 'btn-submit', 'btn-clear', 'alert',
    'routing', 'routing-advice', 'routing-priority',
    'api-details', 'api-request', 'api-response',
    'category-empty', 'category-body', 'verdict-label', 'verdict-confidence',
    'fact-model', 'fact-elapsed',
    'topk-title', 'topk-empty', 'topk-list',
    'sentiment', 'sentiment-label', 'sentiment-note',
    'toast', 'info-dialog', 'info-dialog-title', 'info-dialog-body', 'info-dialog-close'
  ];
  ids.forEach(function (id) {
    dom[id] = document.getElementById(id);
  });
}

/* ==========================================================================
   通用工具
   ========================================================================== */

function setText(node, text) {
  if (node) node.textContent = text;
}

function el(tag, className, text) {
  var node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function clearChildren(node) {
  while (node && node.firstChild) node.removeChild(node.firstChild);
}

function maxReviewChars() {
  var value = dom.review ? dom.review.maxLength : NaN;
  return isFinite(value) && value > 0 ? value : 2000;
}

/** 按 Unicode 码点计数，与后端 len() 的口径一致（而非 UTF-16 码元）。 */
function codePointLength(text) {
  return Array.from(text).length;
}

var toastTimer = null;

function showToast(text) {
  if (!dom.toast) return;
  setText(dom.toast, text);
  dom.toast.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(function () {
    dom.toast.hidden = true;
  }, TOAST_DURATION_MS);
}

function showAlert(text) {
  if (!dom.alert) return;
  setText(dom.alert, text);
  dom.alert.hidden = false;
}

function hideAlert() {
  if (!dom.alert) return;
  setText(dom.alert, '');
  dom.alert.hidden = true;
}

function openDialog(title, nodes) {
  if (!dom['info-dialog']) return;
  setText(dom['info-dialog-title'], title);
  clearChildren(dom['info-dialog-body']);
  nodes.forEach(function (node) {
    dom['info-dialog-body'].appendChild(node);
  });
  if (typeof dom['info-dialog'].showModal === 'function') {
    dom['info-dialog'].showModal();
  } else {
    dom['info-dialog'].setAttribute('open', '');
  }
}

/* ==========================================================================
   服务状态
   ========================================================================== */

function renderServiceStatus(kind) {
  var pill = dom['service-status'];
  var label = { ok: '服务正常', degraded: '模型未就绪', error: '服务异常', checking: '检查中' };
  pill.className = 'pill pill--' + kind;
  setText(dom['service-status-text'], label[kind] || label.error);
  // 状态不只靠颜色表达，文字已同步更新。
  pill.setAttribute('title', label[kind] || '');
}

/* ==========================================================================
   请求
   ========================================================================== */

/** 带超时的 fetch；返回 {status, body, networkError}。非 2xx 不抛异常。 */
function requestJson(url, options, timeoutMs) {
  var controller = typeof AbortController === 'function' ? new AbortController() : null;
  var timer = null;
  var init = Object.assign({}, options || {});
  if (controller) init.signal = controller.signal;

  if (controller) {
    timer = setTimeout(function () { controller.abort(); }, timeoutMs || REQUEST_TIMEOUT_MS);
  }

  return fetch(url, init)
    .then(function (response) {
      return response.json()
        .catch(function () { return null; })
        .then(function (body) {
          return { status: response.status, body: body, networkError: false };
        });
    })
    .catch(function (error) {
      return { status: 0, body: null, networkError: true, error: error };
    })
    .then(function (result) {
      if (timer) clearTimeout(timer);
      return result;
    });
}

function loadHealth() {
  return requestJson(ENDPOINTS.health, { method: 'GET' }).then(function (result) {
    if (result.networkError) {
      renderServiceStatus('error');
      return;
    }
    var kind = classifyHealth(result.status, result.body);
    renderServiceStatus(kind);
    if (kind === 'degraded') {
      // 契约 §3.2：模型未加载时返回 503，属可预期的降级状态，不是故障。
      showAlert('模型尚未加载：可以浏览页面，但预测会返回 MODEL_NOT_LOADED。');
    } else if (kind === 'ok') {
      hideAlert();
    }
  });
}

function loadCategories() {
  return requestJson(ENDPOINTS.categories, { method: 'GET' }).then(function (result) {
    if (result.networkError || !result.body || !Array.isArray(result.body.categories)) return;
    // 类别以接口为准；未收录的类别在建议映射里走通用回退，不另建一份列表。
    window.WebApp.categories = result.body.categories.slice();
  });
}

/* ==========================================================================
   结果渲染
   ========================================================================== */

function resetResult() {
  if (dom['category-empty']) dom['category-empty'].hidden = false;
  if (dom['category-body']) dom['category-body'].hidden = true;
  setText(dom['verdict-label'], '—');
  setText(dom['verdict-confidence'], '—');
  setText(dom['fact-model'], '—');
  setText(dom['fact-elapsed'], '—');

  clearChildren(dom['topk-list']);
  if (dom['topk-empty']) dom['topk-empty'].hidden = false;
  setText(dom['topk-title'], 'Top-' + (dom['top-k'] ? dom['top-k'].value : 3) + ' 概率');

  dom.routing.className = 'routing routing--empty';
  setText(dom['routing-advice'], '分析后给出建议');
  if (dom['routing-priority']) dom['routing-priority'].hidden = true;

  renderSentiment(null);

  setText(dom['api-request'], '暂无请求');
  setText(dom['api-response'], '暂无响应');
  if (dom['api-details']) dom['api-details'].open = false;
}

function renderTopK(items, k) {
  clearChildren(dom['topk-list']);
  setText(dom['topk-title'], 'Top-' + k + ' 概率');
  dom['topk-empty'].hidden = items.length > 0;

  items.forEach(function (item, index) {
    var li = el('li', 'bar' + (index === 0 ? ' bar--top' : ''));

    var label = el('span', 'bar__label', item.label);
    var track = el('span', 'bar__track');
    var fill = el('span', 'bar__fill');
    fill.style.width = (clampRatio(item.probability) * 100).toFixed(2) + '%';
    track.appendChild(fill);
    var value = el('span', 'bar__value', formatPercent(item.probability));

    li.appendChild(label);
    li.appendChild(track);
    li.appendChild(value);
    dom['topk-list'].appendChild(li);
  });
}

function renderSentiment(sentiment) {
  var box = dom.sentiment;
  if (!sentiment) {
    box.className = 'sentiment sentiment--off';
    setText(dom['sentiment-label'], '尚未启用');
    setText(dom['sentiment-note'], '完成 M5 后开放');
    return;
  }
  var negative = sentiment.label === '负向';
  box.className = 'sentiment sentiment--' + (negative ? 'negative' : 'positive');
  setText(dom['sentiment-label'], sentiment.label);
  setText(dom['sentiment-note'], '置信度 ' + formatPercent(sentiment.confidence));
}

function renderRouting(categoryLabel, sentiment) {
  var advice = getRoutingAdvice(categoryLabel, sentiment);
  dom.routing.className = 'routing ' + (advice.urgent ? 'routing--urgent' : 'routing--advised');
  setText(dom['routing-advice'], advice.advice);
  setText(dom['routing-priority'], advice.priority);
  dom['routing-priority'].hidden = false;
}

function renderApiDetails(requestBody, responseBody) {
  var requestSummary = {
    method: 'POST',
    path: ENDPOINTS.predict,
    body: {
      review: requestBody.review.length > 60
        ? requestBody.review.slice(0, 60) + '…'
        : requestBody.review,
      top_k: requestBody.top_k
    }
  };
  setText(dom['api-request'], JSON.stringify(requestSummary, null, 2));
  setText(dom['api-response'], JSON.stringify(responseBody, null, 2));
  if (dom['api-details']) dom['api-details'].open = false;
}

/* ==========================================================================
   提交流程
   ========================================================================== */

var submitting = false;

function setSubmitting(value) {
  submitting = value;
  dom['btn-submit'].disabled = value;
  setText(dom['btn-submit'], value ? '分析中…' : '开始分析');
  // 输入区与 Top-K 在请求期间保持可见可用，仅锁住重复提交。
}

function extractError(body) {
  if (body && body.error && typeof body.error === 'object') {
    return {
      code: typeof body.error.code === 'string' ? body.error.code : '',
      message: typeof body.error.message === 'string' ? body.error.message : ''
    };
  }
  return { code: '', message: '' };
}

function handleSubmit(event) {
  if (event) event.preventDefault();
  if (submitting) return;

  var review = dom.review.value;
  var topK = parseInt(dom['top-k'].value, 10);
  var limit = maxReviewChars();

  hideAlert();

  if (!review.trim()) {
    resetResult();
    showAlert('评论内容不能为空');
    dom.review.focus();
    return;
  }
  if (codePointLength(review) > limit) {
    resetResult();
    showAlert('评论超过 ' + limit + ' 字符，请精简后重试');
    dom.review.focus();
    return;
  }
  if (!(topK >= 1 && topK <= 10)) {
    resetResult();
    showAlert('Top-K 必须是 1 到 10 的整数');
    dom['top-k'].focus();
    return;
  }

  setSubmitting(true);
  resetResult(); // 新请求开始即清除上一次结果，避免误认旧结果

  var payload = { review: review, top_k: topK };

  requestJson(ENDPOINTS.predict, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).then(function (result) {
    if (result.networkError) {
      showAlert('无法连接服务，请确认后端仍在运行后重试');
      renderServiceStatus('error');
      return;
    }

    if (result.status === 200) {
      var check = validatePredictResponse(result.body);
      if (!check.ok) {
        showAlert('服务响应格式异常（' + check.reason + '）');
        return;
      }
      renderSuccess(payload, result.body);
      return;
    }

    var error = extractError(result.body);
    showAlert(mapErrorText(error.code, error.message));
    if (error.code === 'MODEL_NOT_LOADED') {
      renderServiceStatus('degraded');
    } else if (error.code === 'EMPTY_REVIEW' || error.code === 'MISSING_REVIEW') {
      dom.review.focus();
    } else if (error.code === 'INVALID_TOP_K') {
      dom['top-k'].focus();
    }
  }).then(function () {
    setSubmitting(false);
  });
}

function renderSuccess(payload, body) {
  setText(dom['verdict-label'], body.category.label);
  setText(dom['verdict-confidence'], formatPercent(body.category.confidence));
  setText(dom['fact-model'], body.model_name);
  setText(dom['fact-elapsed'], formatElapsedMs(body.elapsed_ms));
  dom['category-empty'].hidden = true;
  dom['category-body'].hidden = false;

  renderTopK(body.top_k, payload.top_k);
  renderSentiment(body.sentiment);
  renderRouting(body.category.label, body.sentiment);
  renderApiDetails(payload, body);
}

/* ==========================================================================
   清空
   ========================================================================== */

function handleClear() {
  dom.review.value = '';
  updateCounter();
  hideAlert();
  resetResult();
  dom.review.focus();
}

function updateCounter() {
  var limit = maxReviewChars();
  var used = codePointLength(dom.review.value);
  setText(dom['char-count'], used + ' / ' + limit);
  dom['char-count'].className = used > limit ? 'counter--over' : '';
}

/* ==========================================================================
   弹窗内容
   ========================================================================== */

function buildHelpNodes() {
  var limit = maxReviewChars();
  return [
    el('h3', null, '输入限制'),
    el('p', null, '单条评论最多 ' + limit + ' 个字符，超出会被后端拒绝并返回 REVIEW_TOO_LONG，' +
      '不会静默截断。空输入或不含任何非空白字符的输入会被拒绝。'),
    el('h3', null, 'Top-K 的含义'),
    el('p', null, 'Top-K 表示返回概率最高的前 K 个类别，取值范围 1 到 10，默认 3。' +
      '概率条按接口返回的顺序展示，第一行即预测类别。'),
    el('h3', null, '常见错误'),
    el('ul', null, null).appendChild(el('li', null, '评论内容不能为空（EMPTY_REVIEW）')),
    el('ul', null, null).appendChild(el('li', null, '模型尚未加载（MODEL_NOT_LOADED）')),
    el('ul', null, null).appendChild(el('li', null, '服务内部错误（INTERNAL_ERROR）'))
  ];
}

function buildAboutNodes() {
  var categories = window.WebApp.categories || [];
  var list = el('ul', null, null);
  categories.forEach(function (name) {
    list.appendChild(el('li', null, name));
  });
  return [
    el('h3', null, '项目任务'),
    el('p', null, '根据中文电商评论文本，预测其所属的商品或服务类别；情感分析为 M5 扩展任务。'),
    el('h3', null, '类别范围（' + categories.length + ' 类）'),
    list,
    el('h3', null, '团队分工'),
    el('ul', null, null).appendChild(el('li', null, 'LanQvJian：项目统筹、工程集成、Web 部署与答辩组织')),
    el('ul', null, null).appendChild(el('li', null, 'HuMingChang：数据工程、EDA、TF-IDF 与传统模型')),
    el('ul', null, null).appendChild(el('li', null, 'HongSenYuan：FastText、BERT、模型优化与性能评估'))
  ];
}

/** 把「多个 ul 各含一个 li」合并成单个列表，避免弹窗里出现一串单点列表。 */
function toSingleList(nodes) {
  var merged = [];
  var current = null;
  nodes.forEach(function (node) {
    if (node.tagName === 'UL') {
      if (!current) {
        current = el('ul', null, null);
        merged.push(current);
      }
      while (node.firstChild) current.appendChild(node.firstChild);
      return;
    }
    current = null;
    merged.push(node);
  });
  return merged;
}

/* ==========================================================================
   启动
   ========================================================================== */

function bindEvents() {
  dom['predict-form'].addEventListener('submit', handleSubmit);
  dom['btn-clear'].addEventListener('click', handleClear);

  dom.review.addEventListener('input', updateCounter);
  dom.review.addEventListener('keydown', function (event) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      handleSubmit(event);
    }
  });

  dom['top-k'].addEventListener('change', function () {
    setText(dom['topk-title'], 'Top-' + dom['top-k'].value + ' 概率');
  });

  dom['btn-help'].addEventListener('click', function () {
    openDialog('帮助文档', toSingleList(buildHelpNodes()));
  });
  dom['btn-about'].addEventListener('click', function () {
    openDialog('关于我们', toSingleList(buildAboutNodes()));
  });
  dom['btn-settings'].addEventListener('click', function () {
    showToast('功能规划中');
  });
  dom['info-dialog-close'].addEventListener('click', function () {
    dom['info-dialog'].close();
  });

  Array.prototype.forEach.call(document.querySelectorAll('[data-planned]'), function (button) {
    button.addEventListener('click', function () {
      showToast('该模块不在本轮 M6 演示范围内');
    });
  });
}

function init() {
  cacheDom();
  bindEvents();
  updateCounter();
  resetResult();
  renderServiceStatus('checking');

  loadHealth();
  loadCategories();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
