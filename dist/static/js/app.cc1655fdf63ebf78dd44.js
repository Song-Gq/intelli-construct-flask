webpackJsonp([1],{"7Otq":function(t,e,a){t.exports=a.p+"static/img/logo.ed32c5d.png"},HQnR:function(t,e){},NHnr:function(t,e,a){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var n=a("7+uW"),i={render:function(){var t=this.$createElement,e=this._self._c||t;return e("div",{staticStyle:{"margin-top":"0"},attrs:{id:"app"}},[e("el-container",[e("el-header",[e("el-row",{staticClass:"row-bg",staticStyle:{height:"100%"},attrs:{type:"flex",justify:"space-between"}},[e("el-menu",{staticClass:"el-menu-demo",staticStyle:{height:"100%"},attrs:{"default-active":this.activeIndex,mode:"horizontal"},on:{select:this.handleSelect}},[e("el-menu-item",{attrs:{index:"1"}},[this._v("Nucleic Acid Test Screenshot OCR")])],1),this._v(" "),e("el-image",{staticStyle:{height:"80%",margin:"auto 10px auto auto"},attrs:{src:a("7Otq"),fit:"scale-down"}})],1)],1),this._v(" "),e("el-main",[e("router-view")],1)],1)],1)},staticRenderFns:[]};var l=a("VU/8")({name:"App",data:function(){return{activeIndex:"1"}},methods:{handleSelect:function(t,e){}},mode:"history"},i,!1,function(t){a("hquX")},null,null).exports,o=a("/ocq"),r=a("mtWM"),s=a.n(r),c={name:"Uploader",data:function(){return{fileList:[],prog:0,in_prog:!1,prog_stat:null,prog_text:"正在上传识别，请耐心等待...",tableData:[],misData:[],timer:null,f_exist:!1,server_available:!1}},computed:{chosenfilenum:function(){return this.fileList.length},resultfilenum:function(){return this.tableData.length}},methods:{handleChange:function(t,e){var a=t.name.lastIndexOf("."),n=t.name.substring(a,t.name.length),i=".jpeg"===n||".jpg"===n||".png"===n,l=t.size/1024/1024<1;i||(this.$message.error("上传图片只能是 JPG/PNG 格式!"),e.pop()),l||(this.$message.error("上传图片大小不能超过 1MB!"),e.pop()),e.length>200&&(this.$message.error("单次识别数量不能超过 200!"),e.pop()),this.fileList=e},submitUpload:function(){var t=this;this.getToken().then(function(e){window.localStorage.setItem("token",e.data),t.server_available=!0,t.in_prog=!0,t.getProgress();var a=new FormData,n=1;t.fileList.forEach(function(t){a.append("id="+n.toString()+"="+t.name,t.raw),n+=1}),t.uploadFile(a).then(function(e){t.prog=100,t.prog_stat="success",t.prog_text="识别成功，刷新页面可重新上传",t.clearTimer();var a=e.data,n=a.res;for(var i in n)t.tableData.push({date:n[i][2],name:n[i][1],type:n[i][0],result:n[i][3]});var l=a.mis;if(null!==l)for(var o in l)t.misData.push({date:l[o][2],name:l[o][1],type:l[o][0],result:l[o][3]})}).catch(function(e){console.log(e),t.$message.error("上传识别失败！"),t.prog_stat="exception",t.prog_text="请刷新页面重试",t.clearTimer()})}).catch(function(e){console.log(e),t.server_available=!1,t.$message.error("服务当前同时使用人数过多！请稍后重试...")})},uploadFile:function(t){return this.$axios.post("http://127.0.0.1:5000/api/recognition",t,{headers:{"Content-Type":"multipart/form-data",token:window.localStorage.getItem("token")}})},getProgress:function(){var t=this;this.timer=setInterval(function(){t.getStatus().then(function(e){-1===e.data?(t.$message.warning("进度获取出现问题...暂不显示实时进度"),t.clearTimer()):(t.prog=Math.round(100*e.data),100===t.prog&&t.clearTimer())}).catch(function(e){t.$message.warning("进度获取出现问题...暂不显示实时进度"),t.clearTimer()})},2e3)},getStatus:function(){return this.$axios.get("http://127.0.0.1:5000/api/getprog",{params:{token:window.localStorage.getItem("token"),timeout:2e3}})},clearTimer:function(){clearInterval(this.timer),this.timer=null},export2excel:function(){this.getExcel().then(function(t){var e=new Blob([t.data],{type:"application/vnd.ms-excel"}),a=document.createElement("a"),n=new Date;a.download="核酸检测报告-"+n.getFullYear()+"-"+n.getMonth()+"-"+n.getDate()+"-"+n.getHours()+"-"+n.getMinutes()+"-"+n.getSeconds()+".xlsx",a.style.display="none",a.href=URL.createObjectURL(e),document.body.appendChild(a),a.click(),URL.revokeObjectURL(a.href),document.body.removeChild(a)})},getExcel:function(){return this.$axios.get("http://127.0.0.1:5000/api/getexcel",{params:{token:window.localStorage.getItem("token")},responseType:"arraybuffer"})},getToken:function(){return this.$axios.get("http://127.0.0.1:5000/api/gettoken",{params:{}})},destroyToken:function(){var t=window.localStorage.getItem("token");if(null!=t)return this.$axios.delete("http://127.0.0.1:5000/api/destroytoken",{params:{token:t}})}},mode:"history",beforeDestroy:function(){clearInterval(this.timer),this.timer=null},beforeMount:function(){window.localStorage.removeItem("token"),this.server_available=!1},mounted:function(){window.addEventListener("beforeunload",function(t){})},created:function(){var t=this;this.$nextTick(function(){t.$refs.upload.$children[0].$refs.input.webkitdirectory=!0})}},p={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticStyle:{"margin-top":"30px"}},[a("div",{directives:[{name:"show",rawName:"v-show",value:t.in_prog,expression:"in_prog"}]},[t._v("\n  "+t._s(t.prog_text)+"\n  "),a("el-progress",{staticStyle:{margin:"5px auto 50px auto",width:"80%"},attrs:{percentage:t.prog,"text-inside":!0,"stroke-width":26,status:t.prog_stat}})],1),t._v(" "),a("el-row",{attrs:{gutter:40}},[a("el-col",{attrs:{span:2}},[a("el-upload",{staticClass:"upload-demo",staticStyle:{float:"left","margin-left":"150%"},attrs:{action:"#",multiple:!0,"auto-upload":!1,"on-change":t.handleChange,"file-list":t.fileList,disabled:t.in_prog,"show-file-list":!1}},[a("el-button",{attrs:{slot:"trigger",size:"small",type:"primary",disabled:t.in_prog},slot:"trigger"},[t._v("选取文件")])],1)],1),t._v(" "),a("el-col",{staticStyle:{"pointer-events":"none"},attrs:{span:8}},[a("el-upload",{ref:"upload",staticClass:"upload",attrs:{action:"#",multiple:!0,"auto-upload":!1,"on-change":t.handleChange,"file-list":t.fileList,disabled:t.in_prog}},[a("el-button",{staticStyle:{"pointer-events":"auto"},attrs:{slot:"trigger",size:"small",type:"primary",disabled:t.in_prog},slot:"trigger"},[t._v("选取文件夹")]),t._v(" "),a("div",{staticClass:"el-upload__tip",staticStyle:{"margin-top":"15px"},attrs:{slot:"tip"},slot:"tip"},[t._v("\n          批量上传核酸检测截图JPEG文件，每张建议不超过200KB")]),t._v(" "),a("div",{staticClass:"el-upload__tip",staticStyle:{"margin-top":"5px"},attrs:{slot:"tip"},slot:"tip"},[t._v("\n          选取文件数："+t._s(t.chosenfilenum))])],1)],1),t._v(" "),a("el-col",{attrs:{span:2}},[0===t.fileList.length?a("el-button",{staticStyle:{float:"right","margin-right":"150%"},attrs:{size:"small",type:"success",disabled:!0}},[t._v("开始识别")]):t._e(),t._v(" "),0!==t.fileList.length?a("el-button",{staticStyle:{float:"right","margin-right":"150%"},attrs:{size:"small",type:"success",disabled:t.in_prog},on:{click:t.submitUpload}},[t._v("开始识别")]):t._e()],1),t._v(" "),a("el-col",{attrs:{span:12}},[[0!==t.misData.length?a("el-result",{staticStyle:{"padding-top":"20px"},attrs:{icon:"warning",title:"提请注意",subTitle:"以下结果请人工复核"}}):t._e(),t._v(" "),0!==t.misData.length?a("el-table",{staticStyle:{width:"100%","margin-bottom":"50px"},attrs:{data:t.misData}},[a("el-table-column",{attrs:{prop:"date",label:"日期",width:"180"}}),t._v(" "),a("el-table-column",{attrs:{prop:"name",label:"姓名",width:"180"}}),t._v(" "),a("el-table-column",{attrs:{prop:"type",label:"类型"}}),t._v(" "),a("el-table-column",{attrs:{prop:"result",label:"结果"}})],1):t._e(),t._v("\n        识别文件数："+t._s(t.resultfilenum)+"\n        "),0!==t.tableData.length?a("el-button",{staticStyle:{margin:"0 auto 20px 50px"},attrs:{size:"small",type:"success"},on:{click:t.export2excel}},[t._v("导出至Excel")]):t._e(),t._v(" "),a("el-table",{staticStyle:{width:"100%","margin-top":"10px"},attrs:{data:t.tableData}},[a("el-table-column",{attrs:{prop:"date",label:"日期",width:"180"}}),t._v(" "),a("el-table-column",{attrs:{prop:"name",label:"姓名",width:"180"}}),t._v(" "),a("el-table-column",{attrs:{prop:"type",label:"类型"}}),t._v(" "),a("el-table-column",{attrs:{prop:"result",label:"结果"}})],1)]],2)],1)],1)},staticRenderFns:[]};var u=a("VU/8")(c,p,!1,function(t){a("HQnR")},"data-v-8d0cd576",null).exports;n.default.use(o.a);var d=new o.a({routes:[{path:"/",name:"Uploader",component:u}]}),g=a("zL8q"),m=a.n(g);a("tvR6");n.default.config.productionTip=!1,n.default.use(m.a),n.default.prototype.$axios=s.a,new n.default({el:"#app",axios:s.a,router:d,components:{App:l},template:"<App/>"})},hquX:function(t,e){},tvR6:function(t,e){}},["NHnr"]);
//# sourceMappingURL=app.cc1655fdf63ebf78dd44.js.map