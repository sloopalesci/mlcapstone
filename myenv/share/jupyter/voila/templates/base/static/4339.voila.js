"use strict";(self.webpackChunk_JUPYTERLAB_CORE_OUTPUT=self.webpackChunk_JUPYTERLAB_CORE_OUTPUT||[]).push([[4339],{44339:(e,t,s)=>{s.r(t),s.d(t,{StyleModule:()=>o});const l="undefined"==typeof Symbol?"__ͼ":Symbol.for("ͼ"),n="undefined"==typeof Symbol?"__styleSet"+Math.floor(1e8*Math.random()):Symbol("styleSet"),i="undefined"!=typeof globalThis?globalThis:"undefined"!=typeof window?window:{};class o{constructor(e,t){this.rules=[];let{finish:s}=t||{};function l(e){return/^@/.test(e)?[e]:e.split(/,\s*/)}function n(e,t,i,o){let r=[],h=/^@(\w+)\b/.exec(e[0]),u=h&&"keyframes"==h[1];if(h&&null==t)return i.push(e[0]+";");for(let s in t){let o=t[s];if(/&/.test(s))n(s.split(/,\s*/).map((t=>e.map((e=>t.replace(/&/,e))))).reduce(((e,t)=>e.concat(t))),o,i);else if(o&&"object"==typeof o){if(!h)throw new RangeError("The value of a property ("+s+") should be a primitive value.");n(l(s),o,r,u)}else null!=o&&r.push(s.replace(/_.*/,"").replace(/[A-Z]/g,(e=>"-"+e.toLowerCase()))+": "+o+";")}(r.length||u)&&i.push((!s||h||o?e:e.map(s)).join(", ")+" {"+r.join(" ")+"}")}for(let t in e)n(l(t),e[t],this.rules)}getRules(){return this.rules.join("\n")}static newName(){let e=i[l]||1;return i[l]=e+1,"ͼ"+e.toString(36)}static mount(e,t,s){let l=e[n],i=s&&s.nonce;l?i&&l.setNonce(i):l=new h(e,i),l.mount(Array.isArray(t)?t:[t])}}let r=new Map;class h{constructor(e,t){let s=e.ownerDocument||e,l=s.defaultView;if(!e.head&&e.adoptedStyleSheets&&l.CSSStyleSheet){let t=r.get(s);if(t)return e.adoptedStyleSheets=[t.sheet,...e.adoptedStyleSheets],e[n]=t;this.sheet=new l.CSSStyleSheet,e.adoptedStyleSheets=[this.sheet,...e.adoptedStyleSheets],r.set(s,this)}else{this.styleTag=s.createElement("style"),t&&this.styleTag.setAttribute("nonce",t);let l=e.head||e;l.insertBefore(this.styleTag,l.firstChild)}this.modules=[],e[n]=this}mount(e){let t=this.sheet,s=0,l=0;for(let n=0;n<e.length;n++){let i=e[n],o=this.modules.indexOf(i);if(o<l&&o>-1&&(this.modules.splice(o,1),l--,o=-1),-1==o){if(this.modules.splice(l++,0,i),t)for(let e=0;e<i.rules.length;e++)t.insertRule(i.rules[e],s++)}else{for(;l<o;)s+=this.modules[l++].rules.length;s+=i.rules.length,l++}}if(!t){let e="";for(let t=0;t<this.modules.length;t++)e+=this.modules[t].getRules()+"\n";this.styleTag.textContent=e}}setNonce(e){this.styleTag&&this.styleTag.getAttribute("nonce")!=e&&this.styleTag.setAttribute("nonce",e)}}}}]);