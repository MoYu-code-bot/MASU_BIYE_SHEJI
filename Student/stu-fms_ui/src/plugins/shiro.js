/*模拟shiro的一个js*/
import request from "@/utils/request"
import { Notification } from 'element-ui'

export default {
   initShiro(vue,callback){
    if(!vue.$store.state.user.id){
      return Notification({
        type:"error",
        message:'用户ID不存在'
      });
    }
    //请求角色数据
     request.get("/api/role/vo/getByUid",{
      params:{
        uid:vue.$store.state.user.id
      }
    }).then(res=>{
      console.log("刷新shiro",res);
      var roles=res.data.data;
      //角色数据
       let roleMap={};
       let permMap={};
      for(var i=0;i<roles.length;i++){
        roleMap[roles[i].roleMark]='role';
        //权限数据
        var prmissions=roles[i].permissions;
        if(prmissions){
          for(var j=0;j<prmissions.length;j++){
            permMap[prmissions[j].perms]=j;
          }
        }
      }
       vue.$store.state.roleMap=roleMap;
       vue.$store.state.permMap=permMap;
      if(callback) callback();
    });
  },
  hasPermission(vue,permission){
    let permMap=vue.$store.state.permMap;
    // console.log("拥有的权限：",permMap);
     if(permMap){
       return permission in permMap;
     }else{
       return false;
     }
  },
  hasRole(vue,role){
     let roleMap=vue.$store.state.roleMap;
    // console.log("拥有的角色：",roleMap);
    if(roleMap){
      return role in roleMap;
    }else{
      return false;
    }
  }
}
