package com.captain.controller;

import com.captain.common.Result;
import com.captain.entity.po.User;
import com.captain.service.UserService;
import org.apache.shiro.SecurityUtils;
import org.apache.shiro.authc.IncorrectCredentialsException;
import org.apache.shiro.authc.UnknownAccountException;
import org.apache.shiro.authc.UsernamePasswordToken;
import org.apache.shiro.authz.annotation.RequiresAuthentication;
import org.apache.shiro.subject.Subject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.Assert;
import org.springframework.web.bind.annotation.*;

/**
 * 用户控制器
 * @author lainhong
 * @date 2020-8-22
 */
@RestController
@RequestMapping("/auth")
public class AuthorController {
    @Autowired
    UserService userService;

    /**
     * 获取当前用户的角色
     * @return
     */
    @GetMapping("/roles")
    public Result getRoles(Long uid){
        return null;
    }

    /**
     * 登录
     * @param user
     * @return
     */
    @PostMapping("/login")
    public Result doLogin(@RequestBody  User user){
        Assert.notNull(user.getUsername(),"登录信息不完整");
        Assert.notNull(user.getPassword(),"登录信息不完整");
        //封装当前用户的登录信息为token
        UsernamePasswordToken token=new UsernamePasswordToken(user.getUsername(),user.getPassword());
        //获取shiro的用户对象
        Subject subject = SecurityUtils.getSubject();
        try {
            //登录这个用户
            subject.login(token);
            //获取这个用户的基本信息
            User loginSuccessUser=(User) subject.getPrincipal();
            return Result.success("登录成功",loginSuccessUser);
        }catch (UnknownAccountException e){
            return Result.fail("用戶不存在");
        }catch (IncorrectCredentialsException e){
            return Result.fail("密码错误");
        }
    }

    /**
     * 退出登录
     * @return
     */
    @RequiresAuthentication
    @PostMapping("/logout")
    public Result logout(){
        Subject subject = SecurityUtils.getSubject();
        subject.logout();
        return Result.success("退出成功",null);
    }

    /**
     * 自助注册（默认赋予 student 角色，与管理员新增用户逻辑一致）
     */
    @PostMapping("/register")
    public Result register(@RequestBody User user) {
        Assert.notNull(user.getUsername(), "用户名不能为空");
        Assert.notNull(user.getPassword(), "密码不能为空");
        String username = user.getUsername().trim();
        Assert.hasText(username, "用户名不能为空");
        user.setUsername(username);
        String err = userService.registerStudentUser(user);
        if (err != null) {
            return Result.fail(err);
        }
        return Result.success("注册成功", null);
    }

}
