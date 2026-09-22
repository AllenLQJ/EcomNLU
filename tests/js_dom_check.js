/**
 * 前端 DOM 行为检查（Node 脚本，**仅测试期使用**）。
 *
 * 前端本身不依赖 Node——它由浏览器直接加载。这里用 Node 的 `vm` 模块跑一个
 * 最小 DOM 垫片，真实执行 `src/web/static/js/app.js` 里的函数并检查产出的
 * 节点树，而不是对源码做字符串断言。`tests/test_web_frontend.py` 调用本脚本；
 * 未安装 Node 时相关用例自动跳过。
 *
 * 用法：
 *   node tests/js_dom_check.js <max_review_chars> <类别1> <类别2> ...
 * 输出：单行 JSON 到 stdout；出错时写 stderr 并以非零码退出。
 */

'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const APP_JS = path.join(__dirname, '..', 'src', 'web', 'static', 'js', 'app.js');
const MAX_REVIEW_CHARS = Number(process.argv[2] || 2000);
const CATEGORIES = process.argv.slice(3);

/** 极简元素垫片：只实现被测代码实际用到的那部分 DOM 接口。 */
function makeElement(tag) {
  const element = {
    tagName: String(tag).toUpperCase(),
    className: '',
    textContent: '',
    value: '',
    maxLength: -1,
    hidden: false,
    style: {},
    attributes: {},
    children: [],
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    removeChild(child) {
      const index = this.children.indexOf(child);
      if (index >= 0) this.children.splice(index, 1);
      return child;
    },
    setAttribute(name, value) {
      this.attributes[name] = value;
    },
    getAttribute(name) {
      return Object.prototype.hasOwnProperty.call(this.attributes, name)
        ? this.attributes[name]
        : null;
    },
    addEventListener() {},
  };
  Object.defineProperty(element, 'firstChild', {
    get() {
      return this.children.length ? this.children[0] : null;
    },
  });
  return element;
}

/** 载入 app.js。把 readyState 设为 loading，使 init() 推迟，不触发启动流程。 */
function boot() {
  const registry = new Map();
  const documentShim = {
    readyState: 'loading',
    createElement: makeElement,
    getElementById(id) {
      if (!registry.has(id)) registry.set(id, makeElement('div'));
      return registry.get(id);
    },
    querySelectorAll() {
      return [];
    },
    addEventListener() {},
  };

  const sandbox = {
    document: documentShim,
    window: {},
    setTimeout() {
      return 0;
    },
    clearTimeout() {},
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(APP_JS, 'utf8'), sandbox, { filename: 'app.js' });
  return { sandbox, registry };
}

/** 复刻 openDialog() 的挂载方式：把节点依次 appendChild 到内容容器。 */
function mount(nodes) {
  const container = makeElement('div');
  nodes.forEach((node) => container.appendChild(node));
  return container;
}

/** 返回标题后紧跟的那个 <ul> 的 <li> 数量；结构不对时返回 null。 */
function listItemsAfter(nodes, heading) {
  for (let i = 0; i < nodes.length; i += 1) {
    if (nodes[i].tagName === 'H3' && nodes[i].textContent === heading) {
      const next = nodes[i + 1];
      if (!next || next.tagName !== 'UL') return null;
      return next.children.filter((child) => child.tagName === 'LI');
    }
  }
  return null;
}

function countDirectLi(container) {
  return container.children.filter((child) => child.tagName === 'LI').length;
}

function main() {
  const { sandbox, registry } = boot();
  const result = {};

  // ---------------- 字符计数器 ----------------
  sandbox.cacheDom();
  const review = registry.get('review');
  const counter = registry.get('char-count');
  review.maxLength = MAX_REVIEW_CHARS;

  const counterStates = {};
  review.value = '';
  sandbox.updateCounter();
  counterStates.initial = { className: counter.className, text: counter.textContent };

  review.value = '手机很好用，电池耐用';
  sandbox.updateCounter();
  counterStates.normal = { className: counter.className, text: counter.textContent };

  review.value = '测'.repeat(MAX_REVIEW_CHARS + 1);
  sandbox.updateCounter();
  counterStates.overLimit = { className: counter.className, text: counter.textContent };

  result.counter = counterStates;

  // ---------------- 弹窗内容 ----------------
  sandbox.window.WebApp.categories = CATEGORIES.slice();

  const helpNodes = sandbox.buildHelpNodes();
  const aboutNodes = sandbox.buildAboutNodes();
  const helpBody = mount(helpNodes);
  const aboutBody = mount(aboutNodes);

  const helpErrors = listItemsAfter(helpNodes, '常见错误');
  const aboutTeam = listItemsAfter(aboutNodes, '团队分工');
  const aboutCategories = listItemsAfter(aboutNodes, '类别范围（' + CATEGORIES.length + ' 类）');

  result.dialog = {
    // 弹窗内容容器下不得直接出现裸 <li>
    directLiUnderContainer: {
      help: countDirectLi(helpBody),
      about: countDirectLi(aboutBody),
    },
    helpErrorListIsUl: helpErrors !== null,
    helpErrorItems: (helpErrors || []).map((item) => item.textContent),
    aboutTeamListIsUl: aboutTeam !== null,
    aboutTeamItems: (aboutTeam || []).map((item) => item.textContent),
    aboutCategoryListIsUl: aboutCategories !== null,
    aboutCategoryItems: (aboutCategories || []).map((item) => item.textContent),
    // 容器下每个子节点的标签名，便于定位结构问题
    helpChildTags: helpBody.children.map((child) => child.tagName),
    aboutChildTags: aboutBody.children.map((child) => child.tagName),
  };

  process.stdout.write(JSON.stringify(result));
}

try {
  main();
} catch (error) {
  process.stderr.write(String((error && error.stack) || error) + '\n');
  process.exit(1);
}
