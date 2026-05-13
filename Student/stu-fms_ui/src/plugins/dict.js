import request from "@/utils/request"

export default {
  initDict(vue,callback){
    //请求角色数据
    request.get("/api/dict/all").then(res=>{
      vue.$store.state.dicts=res.data.data;
      if(callback) callback();
    });
  },
  getValue(vue,key,code){
    if(code){
      let dicts=null;
      try {
        dicts=vue.$store.state.dicts
      }catch (e) {
        //使用表格的scope访问时
        dicts=vue._self.$store.state.dicts;
      }
      if(dicts){
        for(var i=0;i<dicts.length;i++){
          if(dicts[i].dictKey==key&&dicts[i].dictCode==code){
            return dicts[i].dictValue;
          }
        }
      }
    }
    return "--";
  },
  getDicts(vue,key){
    let dicts=vue.$store.state.dicts;
    var array=[];
    if(dicts){
      for(var i=0;i<dicts.length;i++){
        if(dicts[i].dictKey==key){
          array.push(dicts[i]);
        }
      }
    }
    return array;
  }
}
