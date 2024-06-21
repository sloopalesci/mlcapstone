var _JUPYTERLAB;(()=>{"use strict";var e,r,t,a,o,n,i,u,l,s,d,f,p,c,v,h,b,y,g,m,w,j,S,k={120:(e,r,t)=>{var a={"./index":()=>t.e(560).then((()=>()=>t(560))),"./extension":()=>t.e(560).then((()=>()=>t(560))),"./style":()=>t.e(610).then((()=>()=>t(610)))},o=(e,r)=>(t.R=r,r=t.o(a,e)?a[e]():Promise.resolve().then((()=>{throw new Error('Module "'+e+'" does not exist in container.')})),t.R=void 0,r),n=(e,r)=>{if(t.S){var a="default",o=t.S[a];if(o&&o!==e)throw new Error("Container initialization failed as it has already been initialized with a different share scope");return t.S[a]=e,t.I(a,r)}};t.d(r,{get:()=>o,init:()=>n})}},E={};function _(e){var r=E[e];if(void 0!==r)return r.exports;var t=E[e]={id:e,exports:{}};return k[e](t,t.exports,_),t.exports}_.m=k,_.c=E,_.n=e=>{var r=e&&e.__esModule?()=>e.default:()=>e;return _.d(r,{a:r}),r},_.d=(e,r)=>{for(var t in r)_.o(r,t)&&!_.o(e,t)&&Object.defineProperty(e,t,{enumerable:!0,get:r[t]})},_.f={},_.e=e=>Promise.all(Object.keys(_.f).reduce(((r,t)=>(_.f[t](e,r),r)),[])),_.u=e=>e+"."+{560:"0bd998d39a58e3ae6b0a",610:"61ab8ac18f794f39b961"}[e]+".js?v="+{560:"0bd998d39a58e3ae6b0a",610:"61ab8ac18f794f39b961"}[e],_.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"==typeof window)return window}}(),_.o=(e,r)=>Object.prototype.hasOwnProperty.call(e,r),e={},r="@voila-dashboards/jupyterlab-preview:",_.l=(t,a,o,n)=>{if(e[t])e[t].push(a);else{var i,u;if(void 0!==o)for(var l=document.getElementsByTagName("script"),s=0;s<l.length;s++){var d=l[s];if(d.getAttribute("src")==t||d.getAttribute("data-webpack")==r+o){i=d;break}}i||(u=!0,(i=document.createElement("script")).charset="utf-8",i.timeout=120,_.nc&&i.setAttribute("nonce",_.nc),i.setAttribute("data-webpack",r+o),i.src=t),e[t]=[a];var f=(r,a)=>{i.onerror=i.onload=null,clearTimeout(p);var o=e[t];if(delete e[t],i.parentNode&&i.parentNode.removeChild(i),o&&o.forEach((e=>e(a))),r)return r(a)},p=setTimeout(f.bind(null,void 0,{type:"timeout",target:i}),12e4);i.onerror=f.bind(null,i.onerror),i.onload=f.bind(null,i.onload),u&&document.head.appendChild(i)}},_.r=e=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},(()=>{_.S={};var e={},r={};_.I=(t,a)=>{a||(a=[]);var o=r[t];if(o||(o=r[t]={}),!(a.indexOf(o)>=0)){if(a.push(o),e[t])return e[t];_.o(_.S,t)||(_.S[t]={});var n=_.S[t],i="@voila-dashboards/jupyterlab-preview",u=[];return"default"===t&&((e,r,t,a)=>{var o=n[e]=n[e]||{},u=o[r];(!u||!u.loaded&&(1!=!u.eager?a:i>u.from))&&(o[r]={get:()=>_.e(560).then((()=>()=>_(560))),from:i,eager:!1})})("@voila-dashboards/jupyterlab-preview","2.3.7"),e[t]=u.length?Promise.all(u).then((()=>e[t]=1)):1}}})(),(()=>{var e;_.g.importScripts&&(e=_.g.location+"");var r=_.g.document;if(!e&&r&&(r.currentScript&&(e=r.currentScript.src),!e)){var t=r.getElementsByTagName("script");if(t.length)for(var a=t.length-1;a>-1&&!e;)e=t[a--].src}if(!e)throw new Error("Automatic publicPath is not supported in this browser");e=e.replace(/#.*$/,"").replace(/\?.*$/,"").replace(/\/[^\/]+$/,"/"),_.p=e})(),t=e=>{var r=e=>e.split(".").map((e=>+e==e?+e:e)),t=/^([^-+]+)?(?:-([^+]+))?(?:\+(.+))?$/.exec(e),a=t[1]?r(t[1]):[];return t[2]&&(a.length++,a.push.apply(a,r(t[2]))),t[3]&&(a.push([]),a.push.apply(a,r(t[3]))),a},a=(e,r)=>{e=t(e),r=t(r);for(var a=0;;){if(a>=e.length)return a<r.length&&"u"!=(typeof r[a])[0];var o=e[a],n=(typeof o)[0];if(a>=r.length)return"u"==n;var i=r[a],u=(typeof i)[0];if(n!=u)return"o"==n&&"n"==u||"s"==u||"u"==n;if("o"!=n&&"u"!=n&&o!=i)return o<i;a++}},o=e=>{var r=e[0],t="";if(1===e.length)return"*";if(r+.5){t+=0==r?">=":-1==r?"<":1==r?"^":2==r?"~":r>0?"=":"!=";for(var a=1,n=1;n<e.length;n++)a--,t+="u"==(typeof(u=e[n]))[0]?"-":(a>0?".":"")+(a=2,u);return t}var i=[];for(n=1;n<e.length;n++){var u=e[n];i.push(0===u?"not("+l()+")":1===u?"("+l()+" || "+l()+")":2===u?i.pop()+" "+i.pop():o(u))}return l();function l(){return i.pop().replace(/^\((.+)\)$/,"$1")}},n=(e,r)=>{if(0 in e){r=t(r);var a=e[0],o=a<0;o&&(a=-a-1);for(var i=0,u=1,l=!0;;u++,i++){var s,d,f=u<e.length?(typeof e[u])[0]:"";if(i>=r.length||"o"==(d=(typeof(s=r[i]))[0]))return!l||("u"==f?u>a&&!o:""==f!=o);if("u"==d){if(!l||"u"!=f)return!1}else if(l)if(f==d)if(u<=a){if(s!=e[u])return!1}else{if(o?s>e[u]:s<e[u])return!1;s!=e[u]&&(l=!1)}else if("s"!=f&&"n"!=f){if(o||u<=a)return!1;l=!1,u--}else{if(u<=a||d<f!=o)return!1;l=!1}else"s"!=f&&"n"!=f&&(l=!1,u--)}}var p=[],c=p.pop.bind(p);for(i=1;i<e.length;i++){var v=e[i];p.push(1==v?c()|c():2==v?c()&c():v?n(v,r):!c())}return!!c()},i=(e,r)=>{var t=_.S[e];if(!t||!_.o(t,r))throw new Error("Shared module "+r+" doesn't exist in shared scope "+e);return t},u=(e,r)=>{var t=e[r];return(r=Object.keys(t).reduce(((e,r)=>!e||a(e,r)?r:e),0))&&t[r]},l=(e,r)=>{var t=e[r];return Object.keys(t).reduce(((e,r)=>!e||!t[e].loaded&&a(e,r)?r:e),0)},s=(e,r,t,a)=>"Unsatisfied version "+t+" from "+(t&&e[r][t].from)+" of shared singleton module "+r+" (required "+o(a)+")",d=(e,r,t,a)=>{var o=l(e,t);return n(a,o)||c(s(e,t,o,a)),h(e[t][o])},f=(e,r,t)=>{var o=e[r];return(r=Object.keys(o).reduce(((e,r)=>!n(t,r)||e&&!a(e,r)?e:r),0))&&o[r]},p=(e,r,t,a)=>{var n=e[t];return"No satisfying version ("+o(a)+") of shared module "+t+" found in shared scope "+r+".\nAvailable versions: "+Object.keys(n).map((e=>e+" from "+n[e].from)).join(", ")},c=e=>{"undefined"!=typeof console&&console.warn&&console.warn(e)},v=(e,r,t,a)=>{c(p(e,r,t,a))},h=e=>(e.loaded=1,e.get()),y=(b=e=>function(r,t,a,o){var n=_.I(r);return n&&n.then?n.then(e.bind(e,r,_.S[r],t,a,o)):e(r,_.S[r],t,a,o)})(((e,r,t,a)=>(i(e,t),h(f(r,t,a)||v(r,e,t,a)||u(r,t))))),g=b(((e,r,t,a)=>(i(e,t),d(r,0,t,a)))),m={},w={88:()=>g("default","@jupyterlab/notebook",[1,4,2,0]),252:()=>g("default","@jupyterlab/coreutils",[1,6,2,0]),292:()=>g("default","@lumino/signaling",[1,2,0,0]),464:()=>g("default","@lumino/coreutils",[1,2,0,0]),512:()=>g("default","react",[1,18,2,0]),532:()=>g("default","@jupyterlab/apputils",[1,4,3,0]),624:()=>g("default","@jupyterlab/application",[1,4,2,0]),652:()=>g("default","@jupyterlab/ui-components",[1,4,2,0]),805:()=>g("default","@jupyterlab/mainmenu",[1,4,2,0]),828:()=>g("default","@jupyterlab/settingregistry",[1,4,2,0]),912:()=>y("default","@jupyterlab/docregistry",[1,4,2,0])},j={560:[88,252,292,464,512,532,624,652,805,828,912]},S={},_.f.consumes=(e,r)=>{_.o(j,e)&&j[e].forEach((e=>{if(_.o(m,e))return r.push(m[e]);if(!S[e]){var t=r=>{m[e]=0,_.m[e]=t=>{delete _.c[e],t.exports=r()}};S[e]=!0;var a=r=>{delete m[e],_.m[e]=t=>{throw delete _.c[e],r}};try{var o=w[e]();o.then?r.push(m[e]=o.then(t).catch(a)):t(o)}catch(e){a(e)}}}))},(()=>{var e={228:0};_.f.j=(r,t)=>{var a=_.o(e,r)?e[r]:void 0;if(0!==a)if(a)t.push(a[2]);else{var o=new Promise(((t,o)=>a=e[r]=[t,o]));t.push(a[2]=o);var n=_.p+_.u(r),i=new Error;_.l(n,(t=>{if(_.o(e,r)&&(0!==(a=e[r])&&(e[r]=void 0),a)){var o=t&&("load"===t.type?"missing":t.type),n=t&&t.target&&t.target.src;i.message="Loading chunk "+r+" failed.\n("+o+": "+n+")",i.name="ChunkLoadError",i.type=o,i.request=n,a[1](i)}}),"chunk-"+r,r)}};var r=(r,t)=>{var a,o,[n,i,u]=t,l=0;if(n.some((r=>0!==e[r]))){for(a in i)_.o(i,a)&&(_.m[a]=i[a]);u&&u(_)}for(r&&r(t);l<n.length;l++)o=n[l],_.o(e,o)&&e[o]&&e[o][0](),e[o]=0},t=self.webpackChunk_voila_dashboards_jupyterlab_preview=self.webpackChunk_voila_dashboards_jupyterlab_preview||[];t.forEach(r.bind(null,0)),t.push=r.bind(null,t.push.bind(t))})(),_.nc=void 0;var P=_(120);(_JUPYTERLAB=void 0===_JUPYTERLAB?{}:_JUPYTERLAB)["@voila-dashboards/jupyterlab-preview"]=P})();