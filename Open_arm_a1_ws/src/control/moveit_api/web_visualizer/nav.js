/**
 * nav.js — Shared navigation bar for all OpenArm dashboard pages.
 *
 * Call injectNav() once per page (or use the auto-init at the bottom).
 * The bar is a fixed 44px strip at the top. Pages must add `padding-top: 44px`
 * (or equivalent margin) to their main content so nothing is hidden behind it.
 *
 * Connection badge: pass a socketio socket via nav.setSocket(socket) after
 * connecting — the badge updates automatically. Pass null to hide it.
 */
(function (global) {
  'use strict';

  // ── CSS ─────────────────────────────────────────────────────────────────
  var CSS = [
    '#oaNav{',
      'position:fixed;top:0;left:0;right:0;z-index:9999;',
      'height:44px;display:flex;align-items:center;gap:0;',
      'background:rgba(16,18,22,0.97);border-bottom:1px solid rgba(255,255,255,0.10);',
      'font:13px/1 -apple-system,"Segoe UI",Roboto,sans-serif;',
      'backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);',
      'padding:0 14px;box-sizing:border-box;user-select:none;',
    '}',
    '#oaNav .oa-logo{',
      'display:flex;align-items:center;gap:8px;margin-right:18px;flex:none;',
      'text-decoration:none;cursor:pointer;',
    '}',
    '#oaNav .oa-logo svg{display:block;width:20px !important;height:20px !important;min-height:20px !important;max-height:20px !important;flex:none;}',
    '#oaNav .oa-logo-text{',
      'font-weight:700;font-size:14px;letter-spacing:0.01em;',
      'background:linear-gradient(135deg,#6aa9f0 0%,#a78cfa 100%);',
      '-webkit-background-clip:text;-webkit-text-fill-color:transparent;',
      'background-clip:text;',
    '}',
    '#oaNav .oa-nav-links{display:flex;align-items:center;gap:2px;}',
    '#oaNav .oa-nav-link{',
      'padding:6px 12px;border-radius:6px;color:#9aa3ad;',
      'text-decoration:none;font-size:13px;font-weight:500;',
      'transition:color .15s,background .15s;white-space:nowrap;',
    '}',
    '#oaNav .oa-nav-link:hover{color:#e8e8e8;background:rgba(255,255,255,0.07);}',
    '#oaNav .oa-nav-link.active{color:#e8e8e8;background:rgba(58,123,213,0.20);}',
    '#oaNav .oa-spacer{flex:1;}',
    '#oaNav .oa-conn-badge{',
      'padding:3px 9px;border-radius:999px;font-size:11px;font-weight:500;',
      'border:1px solid rgba(255,255,255,0.15);color:#9aa3ad;',
      'transition:color .3s,border-color .3s;white-space:nowrap;',
    '}',
    '#oaNav .oa-conn-badge.live{color:#2ecc71;border-color:#2ecc71;}',
    '#oaNav .oa-conn-badge.dead{color:#d9534f;border-color:#d9534f;}',

    /* Nudge page content below the nav bar on pages that use it */
    'body.oa-nav-active{padding-top:44px !important;}',
  ].join('');

  // ── nav link config ───────────────────────────────────────────────────
  var LINKS = [
    { href: '/3d',     label: '3D' },
    { href: '/fsm',    label: 'FSM' },
    { href: '/logs',   label: 'Logs' },
    { href: '/health', label: 'Health' },
    { href: '/api',    label: 'API' },
  ];

  // ── helpers ────────────────────────────────────────────────────────────

  function currentPage() {
    var p = window.location.pathname;
    if (p.endsWith('/') && p.length > 1) return p.slice(0, -1);
    return p;
  }

  function isActive(href) {
    var cp = currentPage();
    if (href === '/3d' && (cp === '/3d' || cp === '/dashboard' || cp === '/dashboard/' || cp === '/dashboard/index.html')) return true;
    if (href === '/fsm' && (cp === '/fsm' || cp === '/dashboard/fsm.html')) return true;
    if (href === '/logs' && (cp === '/logs' || cp === '/dashboard/logs.html')) return true;
    if (href === '/health' && cp.startsWith('/health')) return true;
    if (href === '/api' && (cp === '/api' || cp.startsWith('/api/docs'))) return true;
    return cp === href;
  }

  // ── build DOM ──────────────────────────────────────────────────────────

  function injectNav(opts) {
    opts = opts || {};
    if (document.getElementById('oaNav')) return; // already injected

    // Inject CSS
    var style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);

    // Build nav bar
    var nav = document.createElement('nav');
    nav.id = 'oaNav';
    nav.setAttribute('role', 'navigation');
    nav.setAttribute('aria-label', 'OpenArm dashboard navigation');

    // Logo (links to Hub /)
    var logo = document.createElement('a');
    logo.href = '/';
    logo.className = 'oa-logo';
    logo.title = 'OpenArm Gateway Hub';
    logo.innerHTML = [
      '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">',
        '<circle cx="10" cy="10" r="9" stroke="#6aa9f0" stroke-width="1.5"/>',
        '<path d="M10 4v6l4 2" stroke="#a78cfa" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>',
        '<circle cx="10" cy="10" r="2" fill="#6aa9f0"/>',
      '</svg>',
      '<span class="oa-logo-text">OpenArm</span>',
    ].join('');
    nav.appendChild(logo);

    // Nav links
    var linksDiv = document.createElement('div');
    linksDiv.className = 'oa-nav-links';
    LINKS.forEach(function (lnk) {
      var a = document.createElement('a');
      a.href = lnk.href;
      a.textContent = lnk.label;
      a.className = 'oa-nav-link' + (isActive(lnk.href) ? ' active' : '');
      // Always open in the same tab
      linksDiv.appendChild(a);
    });
    nav.appendChild(linksDiv);

    // Spacer
    var spacer = document.createElement('div');
    spacer.className = 'oa-spacer';
    nav.appendChild(spacer);

    // Connection badge (hidden by default, shown by setSocket())
    var badge = document.createElement('span');
    badge.className = 'oa-conn-badge';
    badge.id = 'oaConnBadge';
    badge.style.display = 'none';
    nav.appendChild(badge);

    document.body.insertBefore(nav, document.body.firstChild);
    document.body.classList.add('oa-nav-active');
  }

  // ── connection badge ───────────────────────────────────────────────────

  function setBadge(ok, text) {
    var badge = document.getElementById('oaConnBadge');
    if (!badge) return;
    badge.style.display = '';
    badge.textContent = text || (ok ? 'live' : 'disconnected');
    badge.className = 'oa-conn-badge ' + (ok ? 'live' : 'dead');
  }

  function setSocket(socket) {
    if (!socket) {
      var b = document.getElementById('oaConnBadge');
      if (b) b.style.display = 'none';
      return;
    }
    setBadge(false, 'connecting…');
    socket.on('connect',    function () { setBadge(true,  'live'); });
    socket.on('disconnect', function () { setBadge(false, 'socket lost'); });
  }

  // ── favicon injection (guarantees all pages & future pages have SOTA favicons) ──
  function injectFavicon() {
    var head = document.head || document.getElementsByTagName('head')[0];
    if (!head) return;

    var existingIcon = head.querySelector('link[rel~="icon"]');
    if (existingIcon) return;

    var icons = [
      { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
      { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/favicon/favicon-32x32.png' },
      { rel: 'icon', type: 'image/png', sizes: '16x16', href: '/favicon/favicon-16x16.png' },
      { rel: 'apple-touch-icon', sizes: '180x180', href: '/apple-touch-icon.png' },
      { rel: 'manifest', href: '/site.webmanifest' }
    ];

    icons.forEach(function (cfg) {
      var link = document.createElement('link');
      link.rel = cfg.rel;
      if (cfg.type) link.type = cfg.type;
      if (cfg.sizes) link.sizes = cfg.sizes;
      link.href = cfg.href;
      head.appendChild(link);
    });
  }

  // ── auto-init ──────────────────────────────────────────────────────────

  function init() {
    injectFavicon();
    injectNav();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // ── public API ─────────────────────────────────────────────────────────

  global.oaNav = {
    injectNav:     injectNav,
    injectFavicon: injectFavicon,
    setSocket:     setSocket,
    setBadge:      setBadge,
  };

}(window));
