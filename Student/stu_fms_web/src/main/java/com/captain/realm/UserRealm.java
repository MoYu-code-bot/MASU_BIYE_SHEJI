package com.captain.realm;

import com.captain.entity.po.Permission;
import com.captain.entity.po.User;
import com.captain.entity.vo.RoleVo;
import com.captain.service.RoleService;
import com.captain.service.UserService;
import org.apache.shiro.SecurityUtils;
import org.apache.shiro.authc.*;
import org.apache.shiro.authz.AuthorizationInfo;
import org.apache.shiro.authz.SimpleAuthorizationInfo;
import org.apache.shiro.realm.AuthorizingRealm;
import org.apache.shiro.subject.PrincipalCollection;
import org.apache.shiro.subject.Subject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.List;

public class UserRealm extends AuthorizingRealm {
    private final Logger logger= LoggerFactory.getLogger(this.getClass());
    @Autowired
    RoleService roleService;
    @Autowired
    UserService userService;

    //权限授予
    @Override
    protected AuthorizationInfo doGetAuthorizationInfo(PrincipalCollection principalCollection) {
        //获取当前的用户信息
        Subject subject = SecurityUtils.getSubject();
        if(subject==null) return null;
        User user= (User) subject.getPrincipal();
        //进行授权
        if(user!=null){
            SimpleAuthorizationInfo authorizationInfo=new SimpleAuthorizationInfo();
            //得到用户所有的角色
            List<RoleVo> roleVos = roleService.getListByUid(user.getId());
            for(RoleVo item:roleVos){
                //赋予角色
                authorizationInfo.addRole(item.getRoleMark());
                //赋予权限
                for(Permission permission:item.getPermissions()){
                    authorizationInfo.addStringPermission(permission.getPerms());
                }
            }
//            logger.info("授予给【{}】,角色：{}",user.getUsername(),authorizationInfo.getRoles());
//            logger.info("授予给【{}】,权限：{}",user.getUsername(),authorizationInfo.getStringPermissions());
            return authorizationInfo;
        }
        return null;
    }

    //登录认证
    @Override
    protected AuthenticationInfo doGetAuthenticationInfo(AuthenticationToken authenticationToken) throws AuthenticationException {
        //得到用户名密码token
        UsernamePasswordToken token=(UsernamePasswordToken)authenticationToken;
        //从数据库获取当前这个用户
        User user = userService.getByUsername(token.getUsername());
        if(user!=null){
            return new SimpleAuthenticationInfo(user,user.getPassword(),"");
        }
        return null;
    }
}
