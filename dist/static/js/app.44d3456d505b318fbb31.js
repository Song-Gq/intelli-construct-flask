webpackJsonp([1],{"Ann/":function(t,e){},NHnr:function(t,e,a){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var o=a("7+uW"),i={render:function(){var t=this,e=t.$createElement,o=t._self._c||e;return o("div",{staticStyle:{margin:"0 auto"},attrs:{id:"app"}},[o("el-container",[o("el-header",{staticStyle:{padding:"0"}},[o("el-menu",{staticClass:"el-menu-demo",staticStyle:{height:"100%"},attrs:{"default-active":t.activeIndex,mode:"horizontal",router:!0,"background-color":"#545c64","text-color":"#fff","active-text-color":"#ffd04b"},on:{select:t.handleSelect}},[o("el-menu-item",{staticStyle:{"font-size":"18px"},attrs:{index:"index"}},[t._v("基于人工智能的建造安全管理创新示范")]),t._v(" "),o("el-menu-item",{staticStyle:{"font-size":"18px"},attrs:{index:"1"}},[t._v("人员识别")]),t._v(" "),o("el-menu-item",{staticStyle:{"font-size":"18px"},attrs:{index:"2"}},[t._v("驾驶室动作")]),t._v(" "),o("el-menu-item",{staticStyle:{"font-size":"18px"},attrs:{index:"excavatorState"}},[t._v("大型设备")]),t._v(" "),o("el-menu-item",{staticStyle:{"font-size":"18px"},attrs:{index:"4"}},[t._v("设备操作")]),t._v(" "),o("el-menu-item",{staticStyle:{"font-size":"18px"},attrs:{index:"5"}},[t._v("声纹识别")]),t._v(" "),o("div",[o("el-image",{staticStyle:{height:"80%",margin:"5px auto auto auto",position:"absolute",right:"5%"},attrs:{src:a("lUYF"),fit:"scale-down"}}),t._v(" "),o("el-image",{staticStyle:{height:"80%",margin:"5px auto auto auto",position:"absolute",right:"1%"},attrs:{src:a("zfvm"),fit:"scale-down"}})],1)],1)],1),t._v(" "),o("el-main",[o("router-view")],1),t._v(" "),o("el-footer",{staticStyle:{"font-size":"12px",color:"grey",height:"60px"}},[o("div",{staticStyle:{"margin-top":"5px",color:"royalblue"}},[t._v("\n          Source Code Repository on\n          "),o("el-link",{staticStyle:{"font-size":"12px",color:"royalblue"},attrs:{href:"https://github.com/Song-Gq/intelli-construct-vue"}},[t._v("\n            https://github.com/Song-Gq/intelli-construct-vue\n          ")])],1),t._v(" "),o("div",{staticStyle:{"margin-top":"5px"}},[t._v("\n          Copyright © 2022 Trusted AI Lab, Shanghai. All Rights Reserved\n        ")])])],1)],1)},staticRenderFns:[]};var n=a("VU/8")({name:"App",data:function(){return{activeIndex:"index",sumPicNum:0}},methods:{handleSelect:function(t,e){},getActiveMenu:function(){this.activeIndex=this.$router.currentRoute.path.substring(1)}},created:function(){this.getActiveMenu()},mounted:function(){}},i,!1,function(t){a("qHPn")},null,null).exports,s=a("/ocq"),r=a("mtWM"),l=a.n(r),c={name:"Uploader",data:function(){return{fileList:[],prog:0,in_prog:!1,prog_stat:null,prog_text:"正在上传文件，速度取决于网络状况，请耐心等待...",tableData:[],misData:[],timer:null,f_exist:!1,server_available:!1,recog_started:!1}},computed:{chosenfilenum:function(){return this.fileList.length},resultfilenum:function(){return this.tableData.length}},methods:{rowStatus:function(t){t.row,t.rowIndex;return{"background-color":"oldlace"}},handleChange:function(t,e){var a=t.name.lastIndexOf("."),o=t.name.substring(a,t.name.length),i=".jpeg"===o||".jpg"===o||".png"===o,n=t.size/1024/1024<1;i||(this.$message.error("上传图片只能是 JPG/PNG 格式!"),e.pop()),n||(this.$message.error("上传文件大小不能超过 1MB!"),e.pop()),e.length>200&&(this.$message.error("单次识别数量不能超过 200!"),e.pop()),this.fileList=e},submitUpload:function(){var t=this;this.getToken().then(function(e){window.sessionStorage.setItem("token",e.data),t.server_available=!0,t.in_prog=!0,t.getProgress();var a=new FormData,o=1;t.fileList.forEach(function(t){a.append("id="+o.toString()+"="+t.name,t.raw),o+=1}),t.uploadFile(a).then(function(e){console.log(e),console.log(e.status),t.prog=100,t.prog_stat="success",t.prog_text="识别成功，刷新页面可重新上传",t.clearTimer();var a=e.data,o=a.res;for(var i in o)t.tableData.push({date:o[i][2],name:o[i][1],type:o[i][0],result:o[i][3]});var n=a.mis;if(null!==n)for(var s in n)t.misData.push({date:n[s][2],name:n[s][1],type:n[s][0],result:n[s][3]})}).catch(function(e){console.log(e),t.$message.error("上传识别失败！若文件总大小超过20MB，请尝试分批上传"),t.prog_stat="exception",t.prog_text="请刷新页面重试",t.clearTimer()})}).catch(function(e){console.log(e),t.server_available=!1,t.$message.error("服务当前同时使用人数过多！请稍后重试...")})},uploadFile:function(t){return this.$axios.post(this.$targetDomain+"/api/recognition",t,{headers:{"Content-Type":"multipart/form-data",token:window.sessionStorage.getItem("token")}})},getProgress:function(){var t=this;this.timer=setInterval(function(){t.getStatus().then(function(e){-2===e.data?t.recog_started||(t.prog_text="正在上传文件，速度取决于网络状况，请耐心等待...",t.prog=0):-1===e.data?(t.$message.warning("进度获取出现问题...暂不显示实时进度"),t.clearTimer(),t.prog_text="仍正在识别，请耐心等待数分钟...如仍无结果请刷新页面重试"):(t.recog_started=!0,t.prog_text="已上传完成，正在识别，请耐心等待...",t.prog=Math.round(100*e.data),100===t.prog&&t.clearTimer())}).catch(function(e){t.$message.warning("进度获取出现问题...暂不显示实时进度"),t.clearTimer()})},2e3)},getStatus:function(){return this.$axios.get(this.$targetDomain+"/api/getprog",{params:{token:window.sessionStorage.getItem("token"),timeout:2e3}})},clearTimer:function(){clearInterval(this.timer),this.timer=null},export2excel:function(){this.getExcel().then(function(t){var e=new Blob([t.data],{type:"application/vnd.ms-excel"}),a=document.createElement("a"),o=new Date;a.download="核酸检测报告-"+o.getFullYear()+"-"+o.getMonth()+"-"+o.getDate()+"-"+o.getHours()+"-"+o.getMinutes()+"-"+o.getSeconds()+".xls",a.style.display="none",a.href=URL.createObjectURL(e),document.body.appendChild(a),a.click(),URL.revokeObjectURL(a.href),document.body.removeChild(a)})},getExcel:function(){return this.$axios.get(this.$targetDomain+"/api/getexcel",{params:{token:window.sessionStorage.getItem("token")},responseType:"arraybuffer"})},getToken:function(){return this.$axios.get(this.$targetDomain+"/api/gettoken",{params:{}})},destroyToken:function(){var t=window.sessionStorage.getItem("token");if(null!=t)return this.$axios.delete(this.$targetDomain+"/api/destroytoken",{params:{token:t}})}},mode:"history",beforeDestroy:function(){clearInterval(this.timer),this.timer=null},beforeMount:function(){window.sessionStorage.removeItem("token"),this.server_available=!1},mounted:function(){window.addEventListener("beforeunload",function(t){})},created:function(){var t=this;this.$nextTick(function(){t.$refs.upload.$children[0].$refs.input.webkitdirectory=!0})}},p={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticStyle:{"margin-top":"30px"}},[a("div",{directives:[{name:"show",rawName:"v-show",value:t.in_prog,expression:"in_prog"}]},[t._v("\n    "+t._s(t.prog_text)+"\n    "),a("el-progress",{staticStyle:{margin:"5px auto 50px auto",width:"80%"},attrs:{percentage:t.prog,"text-inside":!0,"stroke-width":26,status:t.prog_stat}})],1),t._v(" "),a("el-row",{attrs:{gutter:40}},[a("el-col",{attrs:{span:2}},[a("el-upload",{staticClass:"upload-demo",staticStyle:{float:"left","margin-left":"150%"},attrs:{action:"#",multiple:!0,"auto-upload":!1,"on-change":t.handleChange,"file-list":t.fileList,disabled:t.in_prog,"show-file-list":!1}},[a("el-button",{staticStyle:{"font-size":"14px"},attrs:{slot:"trigger",size:"small",type:"primary",disabled:t.in_prog},slot:"trigger"},[t._v("\n            选取文件")])],1)],1),t._v(" "),a("el-col",{staticStyle:{"pointer-events":"none"},attrs:{span:8}},[a("el-upload",{ref:"upload",staticClass:"upload",attrs:{action:"#",multiple:!0,"auto-upload":!1,"on-change":t.handleChange,"file-list":t.fileList,disabled:t.in_prog}},[a("el-button",{staticStyle:{"pointer-events":"auto","font-size":"14px"},attrs:{slot:"trigger",size:"small",type:"primary",disabled:t.in_prog},slot:"trigger"},[t._v("选取文件夹")]),t._v(" "),a("div",{staticClass:"el-upload__tip",staticStyle:{"margin-top":"5px","font-size":"14px"},attrs:{slot:"tip"},slot:"tip"},[t._v("\n            文件总大小不能超过20MB")]),t._v(" "),a("div",{staticClass:"el-upload__tip",staticStyle:{"margin-top":"5px","font-size":"14px"},attrs:{slot:"tip"},slot:"tip"},[t._v("\n            选取文件数："+t._s(t.chosenfilenum))])],1)],1),t._v(" "),a("el-col",{attrs:{span:2}},[0===t.fileList.length?a("el-button",{staticStyle:{float:"right","margin-right":"150%","font-size":"14px"},attrs:{size:"small",type:"success",disabled:!0}},[t._v("开始识别")]):t._e(),t._v(" "),0!==t.fileList.length?a("el-button",{staticStyle:{float:"right","margin-right":"150%","font-size":"14px"},attrs:{size:"small",type:"success",disabled:t.in_prog},on:{click:t.submitUpload}},[t._v("开始识别")]):t._e()],1),t._v(" "),a("el-col",{attrs:{span:12}},[[0!==t.misData.length?a("el-result",{staticStyle:{"padding-top":"20px"},attrs:{icon:"warning",title:"提请注意",subTitle:"以下结果请人工复核"}}):t._e(),t._v(" "),0!==t.misData.length?a("el-table",{staticStyle:{width:"100%","margin-bottom":"50px"},attrs:{data:t.misData,"row-style":t.rowStatus}},[a("el-table-column",{attrs:{prop:"date",label:"日期"}}),t._v(" "),a("el-table-column",{attrs:{prop:"name",label:"姓名"}}),t._v(" "),a("el-table-column",{attrs:{prop:"type",label:"类型"}}),t._v(" "),a("el-table-column",{attrs:{prop:"result",label:"结果"}})],1):t._e(),t._v(" "),a("div",{staticStyle:{"font-size":"14px"}},[t._v("\n            识别文件数："+t._s(t.resultfilenum)+"\n          ")]),t._v(" "),0!==t.tableData.length?a("el-button",{staticStyle:{margin:"20px auto 20px auto"},attrs:{size:"small",type:"success"},on:{click:t.export2excel}},[t._v("导出至Excel")]):t._e(),t._v(" "),a("el-table",{staticStyle:{width:"100%","margin-top":"10px"},attrs:{data:t.tableData,stripe:!0,"max-height":800,size:"small"}},[a("el-table-column",{attrs:{prop:"date",label:"日期"}}),t._v(" "),a("el-table-column",{attrs:{prop:"name",label:"姓名"}}),t._v(" "),a("el-table-column",{attrs:{prop:"type",label:"类型"}}),t._v(" "),a("el-table-column",{attrs:{prop:"result",label:"结果"}})],1)]],2)],1),t._v(" "),a("iframe",{staticStyle:{"margin-top":"40px"},attrs:{width:"1120",height:"630",src:"//player.bilibili.com/player.html?aid=764029001&bvid=BV1Ur4y1C73M&cid=438374904&page=1",scrolling:"no",border:"0",frameborder:"no",framespacing:"0",allowfullscreen:"true"}})],1)},staticRenderFns:[]};a("VU/8")(c,p,!1,function(t){a("sDrd")},"data-v-f408862a",null).exports;var u={name:"Navi",data:function(){return{fileList:[],prog:0,in_prog:!1,prog_stat:null,prog_text:"正在上传文件，速度取决于网络状况，请耐心等待...",tableData:[],misData:[],timer:null,f_exist:!1,server_available:!1,recog_started:!1}},computed:{chosenfilenum:function(){return this.fileList.length},resultfilenum:function(){return this.tableData.length}},methods:{rowStatus:function(t){t.row,t.rowIndex;return{"background-color":"oldlace"}},handleChange:function(t,e){var a=t.name.lastIndexOf("."),o=t.name.substring(a,t.name.length),i=".jpeg"===o||".jpg"===o||".png"===o,n=t.size/1024/1024<1;i||(this.$message.error("上传图片只能是 JPG/PNG 格式!"),e.pop()),n||(this.$message.error("上传文件大小不能超过 1MB!"),e.pop()),e.length>200&&(this.$message.error("单次识别数量不能超过 200!"),e.pop()),this.fileList=e},submitUpload:function(){var t=this;this.getToken().then(function(e){window.sessionStorage.setItem("token",e.data),t.server_available=!0,t.in_prog=!0,t.getProgress();var a=new FormData,o=1;t.fileList.forEach(function(t){a.append("id="+o.toString()+"="+t.name,t.raw),o+=1}),t.uploadFile(a).then(function(e){console.log(e),console.log(e.status),t.prog=100,t.prog_stat="success",t.prog_text="识别成功，刷新页面可重新上传",t.clearTimer();var a=e.data,o=a.res;for(var i in o)t.tableData.push({date:o[i][2],name:o[i][1],type:o[i][0],result:o[i][3]});var n=a.mis;if(null!==n)for(var s in n)t.misData.push({date:n[s][2],name:n[s][1],type:n[s][0],result:n[s][3]})}).catch(function(e){console.log(e),t.$message.error("上传识别失败！若文件总大小超过20MB，请尝试分批上传"),t.prog_stat="exception",t.prog_text="请刷新页面重试",t.clearTimer()})}).catch(function(e){console.log(e),t.server_available=!1,t.$message.error("服务当前同时使用人数过多！请稍后重试...")})},uploadFile:function(t){return this.$axios.post(this.$targetDomain+"/api/recognition",t,{headers:{"Content-Type":"multipart/form-data",token:window.sessionStorage.getItem("token")}})},getProgress:function(){var t=this;this.timer=setInterval(function(){t.getStatus().then(function(e){-2===e.data?t.recog_started||(t.prog_text="正在上传文件，速度取决于网络状况，请耐心等待...",t.prog=0):-1===e.data?(t.$message.warning("进度获取出现问题...暂不显示实时进度"),t.clearTimer(),t.prog_text="仍正在识别，请耐心等待数分钟...如仍无结果请刷新页面重试"):(t.recog_started=!0,t.prog_text="已上传完成，正在识别，请耐心等待...",t.prog=Math.round(100*e.data),100===t.prog&&t.clearTimer())}).catch(function(e){t.$message.warning("进度获取出现问题...暂不显示实时进度"),t.clearTimer()})},2e3)},getStatus:function(){return this.$axios.get(this.$targetDomain+"/api/getprog",{params:{token:window.sessionStorage.getItem("token"),timeout:2e3}})},clearTimer:function(){clearInterval(this.timer),this.timer=null},export2excel:function(){this.getExcel().then(function(t){var e=new Blob([t.data],{type:"application/vnd.ms-excel"}),a=document.createElement("a"),o=new Date;a.download="核酸检测报告-"+o.getFullYear()+"-"+o.getMonth()+"-"+o.getDate()+"-"+o.getHours()+"-"+o.getMinutes()+"-"+o.getSeconds()+".xls",a.style.display="none",a.href=URL.createObjectURL(e),document.body.appendChild(a),a.click(),URL.revokeObjectURL(a.href),document.body.removeChild(a)})},getExcel:function(){return this.$axios.get(this.$targetDomain+"/api/getexcel",{params:{token:window.sessionStorage.getItem("token")},responseType:"arraybuffer"})},getToken:function(){return this.$axios.get(this.$targetDomain+"/api/gettoken",{params:{}})},destroyToken:function(){var t=window.sessionStorage.getItem("token");if(null!=t)return this.$axios.delete(this.$targetDomain+"/api/destroytoken",{params:{token:t}})}},mode:"history",beforeDestroy:function(){clearInterval(this.timer),this.timer=null},beforeMount:function(){window.sessionStorage.removeItem("token"),this.server_available=!1},mounted:function(){window.addEventListener("beforeunload",function(t){})},created:function(){var t=this;this.$nextTick(function(){t.$refs.upload.$children[0].$refs.input.webkitdirectory=!0})}},g={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticStyle:{"margin-top":"30px"}},[a("div",{directives:[{name:"show",rawName:"v-show",value:t.in_prog,expression:"in_prog"}]},[t._v("\n      "+t._s(t.prog_text)+"\n      "),a("el-progress",{staticStyle:{margin:"5px auto 50px auto",width:"80%"},attrs:{percentage:t.prog,"text-inside":!0,"stroke-width":26,status:t.prog_stat}})],1),t._v(" "),a("el-row",{attrs:{gutter:40}},[a("el-col",{attrs:{span:2}},[a("el-upload",{staticClass:"upload-demo",staticStyle:{float:"left","margin-left":"150%"},attrs:{action:"#",multiple:!0,"auto-upload":!1,"on-change":t.handleChange,"file-list":t.fileList,disabled:t.in_prog,"show-file-list":!1}})],1),t._v(" "),a("el-col",{staticStyle:{"pointer-events":"none"},attrs:{span:8}},[a("el-upload",{ref:"upload",staticClass:"upload",attrs:{action:"#",multiple:!0,"auto-upload":!1,"on-change":t.handleChange,"file-list":t.fileList,disabled:t.in_prog}},[a("div",{staticClass:"el-upload__tip",staticStyle:{"margin-top":"5px","font-size":"14px"},attrs:{slot:"tip"},slot:"tip"},[t._v("\n            文件总大小不能超过20MB")]),t._v(" "),a("div",{staticClass:"el-upload__tip",staticStyle:{"margin-top":"5px","font-size":"14px"},attrs:{slot:"tip"},slot:"tip"},[t._v("\n            选取文件数："+t._s(t.chosenfilenum))])])],1),t._v(" "),a("el-col",{attrs:{span:2}},[0===t.fileList.length?a("el-button",{staticStyle:{float:"right","margin-right":"150%","font-size":"14px"},attrs:{size:"small",type:"success",disabled:!0}},[t._v("开始识别")]):t._e(),t._v(" "),0!==t.fileList.length?a("el-button",{staticStyle:{float:"right","margin-right":"150%","font-size":"14px"},attrs:{size:"small",type:"success",disabled:t.in_prog},on:{click:t.submitUpload}},[t._v("开始识别")]):t._e()],1),t._v(" "),a("el-col",{attrs:{span:12}},[[0!==t.misData.length?a("el-result",{staticStyle:{"padding-top":"20px"},attrs:{icon:"warning",title:"提请注意",subTitle:"以下结果请人工复核"}}):t._e(),t._v(" "),0!==t.misData.length?a("el-table",{staticStyle:{width:"100%","margin-bottom":"50px"},attrs:{data:t.misData,"row-style":t.rowStatus}},[a("el-table-column",{attrs:{prop:"date",label:"日期"}}),t._v(" "),a("el-table-column",{attrs:{prop:"name",label:"姓名"}}),t._v(" "),a("el-table-column",{attrs:{prop:"type",label:"类型"}}),t._v(" "),a("el-table-column",{attrs:{prop:"result",label:"结果"}})],1):t._e(),t._v(" "),a("div",{staticStyle:{"font-size":"14px"}},[t._v("\n            识别文件数："+t._s(t.resultfilenum)+"\n          ")]),t._v(" "),0!==t.tableData.length?a("el-button",{staticStyle:{margin:"20px auto 20px auto"},attrs:{size:"small",type:"success"},on:{click:t.export2excel}},[t._v("导出至Excel")]):t._e(),t._v(" "),a("el-table",{staticStyle:{width:"100%","margin-top":"10px"},attrs:{data:t.tableData,stripe:!0,"max-height":800,size:"small"}},[a("el-table-column",{attrs:{prop:"date",label:"日期"}}),t._v(" "),a("el-table-column",{attrs:{prop:"name",label:"姓名"}}),t._v(" "),a("el-table-column",{attrs:{prop:"type",label:"类型"}}),t._v(" "),a("el-table-column",{attrs:{prop:"result",label:"结果"}})],1)]],2)],1),t._v(" "),a("iframe",{staticStyle:{"margin-top":"40px"},attrs:{width:"1120",height:"630",src:"//player.bilibili.com/player.html?aid=764029001&bvid=BV1Ur4y1C73M&cid=438374904&page=1",scrolling:"no",border:"0",frameborder:"no",framespacing:"0",allowfullscreen:"true"}})],1)},staticRenderFns:[]};var d=a("VU/8")(u,g,!1,function(t){a("Ann/")},"data-v-9eb1007a",null).exports,m=a("KkKn"),h={props:["figData"],components:{Plotly:m.Plotly},name:"Loc2d",data:function(){return{layout:{title:"excavator"}}}},f={render:function(){var t=this.$createElement;return(this._self._c||t)("Plotly",{attrs:{data:this.figData,layout:this.layout,"display-mode-bar":!1}},[this._v("}")])},staticRenderFns:[]};var v={components:{Loc2d:a("VU/8")(h,f,!1,function(t){a("sA29")},"data-v-5ccf21e6",null).exports},name:"ExcavatorState",data:function(){return{fileList:[],prog:0,in_prog:!1,prog_stat:null,prog_text:"正在上传文件，速度取决于网络状况，请耐心等待...",tableData:[],misData:[],timer:null,f_exist:!1,server_available:!1,recog_started:!1,areaFig:[{x:[],y:[],type:"scatter"}],posFig:[{x:[],y:[],type:"scatter"},{x:[],y:[],type:"scatter"}]}},computed:{chosenfilenum:function(){return this.fileList.length},resultfilenum:function(){return this.tableData.length}},methods:{rowStatus:function(t){t.row,t.rowIndex;return{"background-color":"oldlace"}},handleChange:function(t,e){var a=t.name.lastIndexOf("."),o=".mp4"===t.name.substring(a,t.name.length).toLowerCase(),i=t.size/1024/1024<10;o||(this.$message.error("上传视频只能是 MP4 格式!"),e.pop()),i||(this.$message.error("上传文件大小不能超过 10MB!"),e.pop()),e.length>200&&(this.$message.error("单次识别数量不能超过 200!"),e.pop()),this.fileList=e},submitUpload:function(){var t=this;this.getToken().then(function(e){window.sessionStorage.setItem("token",e.data),t.server_available=!0,t.in_prog=!0,t.getProgress();var a=new FormData,o=1;t.fileList.forEach(function(t){a.append("id="+o.toString()+"="+t.name,t.raw),o+=1}),t.uploadFile(a).then(function(e){console.log(e),console.log(e.status),t.prog=100,t.prog_stat="success",t.prog_text="识别成功，刷新页面可重新上传",t.clearTimer();var a=e.data,o=a.res;for(var i in o){t.tableData.push({frame:o[i][0],excav:o[i][1],top:o[i][2],left:o[i][3],width:o[i][4],height:o[i][5],area:o[i][6]});var n=o[i][3]+o[i][4]/2,s=o[i][2]+o[i][5]/2;t.areaFig[0].x.push(o[i][0]),t.areaFig[0].y.push(o[i][6]),t.posFig[0].x.push(o[i][0]),t.posFig[0].y.push(n),t.posFig[1].x.push(o[i][0]),t.posFig[1].y.push(s)}console.log(t.posFig);var r=a.mis;if(null!==r)for(var l in r)t.misData.push({date:r[l][2],name:r[l][1],type:r[l][0],result:r[l][3]})}).catch(function(e){console.log(e),t.$message.error("上传识别失败！"),t.prog_stat="exception",t.prog_text="请刷新页面重试",t.clearTimer()})}).catch(function(e){console.log(e),t.server_available=!1,t.$message.error("服务当前同时使用人数过多！请稍后重试...")})},uploadFile:function(t){return this.$axios.post(this.$targetDomain+"/api/excavator",t,{headers:{"Content-Type":"multipart/form-data",token:window.sessionStorage.getItem("token")}})},getProgress:function(){var t=this;this.timer=setInterval(function(){t.getStatus().then(function(e){-2===e.data?t.recog_started||(t.prog_text="正在上传文件，速度取决于网络状况，请耐心等待...",t.prog=0):-1===e.data?(t.$message.warning("进度获取出现问题...暂不显示实时进度"),t.clearTimer(),t.prog_text="仍正在识别，请耐心等待数分钟...如仍无结果请刷新页面重试"):(t.recog_started=!0,t.prog_text="已上传完成，正在识别，请耐心等待...",t.prog=Math.round(100*e.data),100===t.prog&&t.clearTimer())}).catch(function(e){t.$message.warning("进度获取出现问题...暂不显示实时进度"),t.clearTimer()})},2e3)},getStatus:function(){return this.$axios.get(this.$targetDomain+"/api/getprog",{params:{token:window.sessionStorage.getItem("token"),timeout:2e3}})},clearTimer:function(){clearInterval(this.timer),this.timer=null},export2excel:function(){this.getExcel().then(function(t){var e=new Blob([t.data],{type:"application/vnd.ms-excel"}),a=document.createElement("a"),o=new Date;a.download="核酸检测报告-"+o.getFullYear()+"-"+o.getMonth()+"-"+o.getDate()+"-"+o.getHours()+"-"+o.getMinutes()+"-"+o.getSeconds()+".xls",a.style.display="none",a.href=URL.createObjectURL(e),document.body.appendChild(a),a.click(),URL.revokeObjectURL(a.href),document.body.removeChild(a)})},getExcel:function(){return this.$axios.get(this.$targetDomain+"/api/getexcel",{params:{token:window.sessionStorage.getItem("token")},responseType:"arraybuffer"})},getToken:function(){return this.$axios.get(this.$targetDomain+"/api/gettoken",{params:{}})},destroyToken:function(){var t=window.sessionStorage.getItem("token");if(null!=t)return this.$axios.delete(this.$targetDomain+"/api/destroytoken",{params:{token:t}})}},mode:"history",beforeDestroy:function(){clearInterval(this.timer),this.timer=null},beforeMount:function(){window.sessionStorage.removeItem("token"),this.server_available=!1},mounted:function(){window.addEventListener("beforeunload",function(t){})},created:function(){var t=this;this.$nextTick(function(){t.$refs.upload.$children[0].$refs.input.webkitdirectory=!0})}},_={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticStyle:{"margin-top":"30px"}},[a("el-row",{attrs:{gutter:40}},[a("el-col",{attrs:{span:1}},[a("el-upload",{staticClass:"upload-demo",staticStyle:{float:"left","margin-left":"150%"},attrs:{action:"#",multiple:!0,"auto-upload":!1,"on-change":t.handleChange,"file-list":t.fileList,disabled:t.in_prog,"show-file-list":!1}},[a("el-button",{staticStyle:{"font-size":"14px"},attrs:{slot:"trigger",size:"small",type:"primary",disabled:t.in_prog},slot:"trigger"},[t._v("\n            选取视频文件")])],1)],1),t._v(" "),a("el-col",{staticStyle:{"pointer-events":"none"},attrs:{span:3}},[a("el-upload",{ref:"upload",staticClass:"upload",attrs:{action:"#",multiple:!0,"auto-upload":!1,"on-change":t.handleChange,"file-list":t.fileList,disabled:t.in_prog}},[a("div",{staticClass:"el-upload__tip",staticStyle:{"margin-top":"30px","font-size":"14px"},attrs:{slot:"tip"},slot:"tip"},[t._v("\n            文件大小不能超过10MB")]),t._v(" "),a("div",{staticClass:"el-upload__tip",staticStyle:{"margin-top":"5px","font-size":"14px"},attrs:{slot:"tip"},slot:"tip"},[t._v("\n            选取文件数："+t._s(t.chosenfilenum))])])],1),t._v(" "),a("el-col",{attrs:{span:1}},[0===t.fileList.length?a("el-button",{staticStyle:{float:"right","margin-right":"150%","font-size":"14px"},attrs:{size:"small",type:"success",disabled:!0}},[t._v("开始识别")]):t._e(),t._v(" "),0!==t.fileList.length?a("el-button",{staticStyle:{float:"right","margin-right":"150%","font-size":"14px"},attrs:{size:"small",type:"success",disabled:t.in_prog},on:{click:t.submitUpload}},[t._v("开始识别")]):t._e()],1),t._v(" "),a("el-col",{attrs:{span:19}},[a("div",{directives:[{name:"show",rawName:"v-show",value:t.in_prog,expression:"in_prog"}]},[t._v("\n          "+t._s(t.prog_text)+"\n          "),a("el-progress",{staticStyle:{margin:"5px auto 50px auto",width:"80%"},attrs:{percentage:t.prog,"text-inside":!0,"stroke-width":26,status:t.prog_stat}})],1)])],1),t._v(" "),a("el-row",{attrs:{gutter:40}},[a("el-col",{attrs:{span:12}},[a("Loc2d",{staticStyle:{height:"800px"},attrs:{figData:t.posFig}}),t._v(" "),a("Loc2d",{staticStyle:{height:"800px"},attrs:{figData:t.areaFig}})],1),t._v(" "),a("el-col",{attrs:{span:12}},[[0!==t.misData.length?a("el-result",{staticStyle:{"padding-top":"20px"},attrs:{icon:"warning",title:"提请注意",subTitle:"以下结果请人工复核"}}):t._e(),t._v(" "),0!==t.misData.length?a("el-table",{staticStyle:{width:"100%","margin-bottom":"50px"},attrs:{data:t.misData,"row-style":t.rowStatus}},[a("el-table-column",{attrs:{prop:"date",label:"日期"}}),t._v(" "),a("el-table-column",{attrs:{prop:"name",label:"姓名"}}),t._v(" "),a("el-table-column",{attrs:{prop:"type",label:"类型"}}),t._v(" "),a("el-table-column",{attrs:{prop:"result",label:"结果"}})],1):t._e(),t._v(" "),a("div",{staticStyle:{"font-size":"14px"}},[t._v("\n            有目标帧数："+t._s(t.resultfilenum)+"\n          ")]),t._v(" "),0!==t.tableData.length?a("el-button",{staticStyle:{margin:"20px auto 20px auto"},attrs:{size:"small",type:"success"},on:{click:t.export2excel}},[t._v("导出至Excel")]):t._e(),t._v(" "),a("el-table",{staticStyle:{width:"100%","margin-top":"10px"},attrs:{data:t.tableData,stripe:!0,"max-height":1500,size:"small"}},[a("el-table-column",{attrs:{prop:"frame",label:"帧"}}),t._v(" "),a("el-table-column",{attrs:{prop:"excav",label:"挖机"}}),t._v(" "),a("el-table-column",{attrs:{prop:"top",label:"顶"}}),t._v(" "),a("el-table-column",{attrs:{prop:"left",label:"左"}}),t._v(" "),a("el-table-column",{attrs:{prop:"width",label:"宽"}}),t._v(" "),a("el-table-column",{attrs:{prop:"height",label:"高"}}),t._v(" "),a("el-table-column",{attrs:{prop:"area",label:"面积"}})],1)]],2)],1)],1)},staticRenderFns:[]};var x=a("VU/8")(v,_,!1,function(t){a("waG/")},"data-v-e83d3b64",null).exports;o.default.use(s.a);var b=new s.a({routes:[{path:"/",redirect:"/index"},{path:"/index",name:"Navi",component:d},{path:"/excavatorState",name:"excavatorState",component:x}],mode:"history"}),y=a("zL8q"),w=a.n(y);a("tvR6");o.default.config.productionTip=!1,o.default.use(w.a),o.default.prototype.$axios=l.a;o.default.prototype.$targetDomain="https://x.gqsong.xyz:443",new o.default({el:"#app",axios:l.a,targetDomain:"https://x.gqsong.xyz:443",router:b,components:{App:n},template:"<App/>"})},lUYF:function(t,e,a){t.exports=a.p+"static/img/tongji-w.9f17702.png"},qHPn:function(t,e){},sA29:function(t,e){},sDrd:function(t,e){},tvR6:function(t,e){},"waG/":function(t,e){},zfvm:function(t,e,a){t.exports=a.p+"static/img/logo-w.7285c59.png"}},["NHnr"]);
//# sourceMappingURL=app.44d3456d505b318fbb31.js.map